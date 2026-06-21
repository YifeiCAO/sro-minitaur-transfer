"""Phase 1 -- M_pop: population fine-tune of the behavioral FM on SRO, no
individual information, then frozen as the no-individual floor baseline.

Why fine-tune before freezing: executive-control tasks are OOD for a Psych-101
model, so a raw frozen model is likely miscalibrated on SRO. If we skip this,
Phase-2's "improvement from z" would partly be the encoder fixing population
miscalibration rather than carrying individual information. M_pop closes that
gap so the shuffled-z control is clean.

Loss is masked to ``<<...>>`` response tokens only (see ``masking.py``).

Requires a GPU (Colab). All heavy imports are local to keep the package light.
"""
from __future__ import annotations

from pathlib import Path

from .masking import build_labels


def load_base_model(cfg: dict):
    """Load the behavioral FM (Centaur 8B) + tokenizer, 4-bit for Colab.

    Default: ``base_model`` is the merged Centaur model, loaded directly. If
    ``base_is_adapter`` is true, ``base_model`` is instead a LoRA adapter applied
    on ``base_llm`` and merged. Either way a *fresh* LoRA is then attached for
    the SRO population fine-tune.
    """
    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    # optional: fused cross-entropy (Liger) avoids materializing the [B,L,V] fp32
    # logits tensor -> the seq-4096 memory bottleneck -> much larger batch fits.
    # No-op if liger-kernel isn't installed.
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        apply_liger_kernel_to_llama()
        print("[liger] fused kernels enabled (raise --batch-size to fill VRAM)")
    except Exception as e:
        print(f"[liger] not active ({type(e).__name__}); standard path (keep batch modest)")

    m = cfg["model"]
    src = m["base_llm"] if m.get("base_is_adapter") else m["base_model"]
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if m.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Only when base_model is a LoRA adapter: apply it on the raw Llama base and
    # merge. Default path loads the merged Centaur directly above, so this is a
    # no-op.
    if m.get("base_is_adapter") and m.get("base_model"):
        model = PeftModel.from_pretrained(model, m["base_model"])
        model = model.merge_and_unload()

    # standard QLoRA prep for 4-bit training (upcasts norms, enables grad flow
    # through the frozen base, turns on gradient checkpointing for memory)
    if m.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # fresh LoRA for the SRO population fine-tune
    ft = cfg["mpop_finetune"]
    lora = LoraConfig(
        r=ft["lora_r"], lora_alpha=ft["lora_alpha"], lora_dropout=ft["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok


def load_mpop(cfg: dict, mpop_dir: str | Path):
    """Reload a trained, FROZEN M_pop: base behavioral FM + the SRO LoRA adapter.

    ``trainer.save_model`` writes only the LoRA adapter, so we must reload the
    base (merged Minitaur by default) and apply the adapter on top. Returns
    (model in eval mode with grads off, tokenizer).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    m = cfg["model"]
    src = m["base_llm"] if m.get("base_is_adapter") else m["base_model"]
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    quant = None
    if m.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(str(mpop_dir))
    base = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=quant, torch_dtype=compute_dtype, device_map="auto",
    )
    if m.get("base_is_adapter") and m.get("base_model"):
        base = PeftModel.from_pretrained(base, m["base_model"]).merge_and_unload()
    model = PeftModel.from_pretrained(base, str(mpop_dir))  # the trained SRO adapter
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def load_for_incontext_finetune(cfg: dict, mpop_dir: str | Path):
    """Load M_pop with its SRO adapter set TRAINABLE, to continue fine-tuning it
    on [A-session + B-session] sequences (teach it to use cross-task context).
    Standard LoRA fine-tune -- no soft-prompt / no backprop-to-input fragility.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    m = cfg["model"]
    if m.get("base_is_adapter"):
        raise NotImplementedError("in-context finetune assumes a merged base_model")
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    quant = None
    if m.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(str(mpop_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        m["base_model"], quantization_config=quant, torch_dtype=compute_dtype, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(mpop_dir), is_trainable=True)
    model.enable_input_require_grads()           # needed for grad checkpointing
    model.print_trainable_parameters()
    return model, tok


def load_raw_model(cfg: dict):
    """Raw base behavioral FM (NO SRO adapter), 4-bit, frozen — for training-free
    surprise reps / scale checks (e.g. Centaur-70B). Handles a merged base_model
    OR an adapter-on-base_llm (set base_is_adapter + base_llm, e.g. gated Llama-70B).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    m = cfg["model"]
    src = m["base_llm"] if m.get("base_is_adapter") else m["base_model"]
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    quant = None
    if m.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(m.get("base_model") or src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=quant, torch_dtype=compute_dtype, device_map="auto",
    )
    if m.get("base_is_adapter") and m.get("base_model"):
        model = PeftModel.from_pretrained(model, m["base_model"]).merge_and_unload()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def build_dataset(sessions: dict[str, str], tokenizer, max_len: int):
    """HF Dataset of response-masked sessions (one row per (subject, task))."""
    import os
    from datasets import Dataset

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # safe under multiprocessing
    rows = [{"text": t} for t in sessions.values()]
    ds = Dataset.from_list(rows)

    def _map(ex):
        return build_labels(ex["text"], tokenizer, max_len)

    nproc = min(8, max(1, (os.cpu_count() or 2)))
    return ds.map(_map, remove_columns=["text"], num_proc=nproc)


def train_mpop(cfg: dict, sessions: dict[str, str], out_dir: str | Path):
    """Run the population fine-tune. ``sessions`` must be TRAIN subjects only."""
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    model, tok = load_base_model(cfg)
    ds = build_dataset(sessions, tok, cfg["model"]["max_seq_len"])
    ft = cfg["mpop_finetune"]
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=ft["batch_size"],
        gradient_accumulation_steps=ft["grad_accum"],
        learning_rate=ft["lr"],
        num_train_epochs=ft["epochs"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps",
        save_steps=ft.get("save_steps", 100),     # checkpoint to Drive ~every 45 min
        save_total_limit=2,                        # keep disk small
        report_to=[],
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    # resume from the latest checkpoint if one exists (survives a Colab disconnect)
    import os
    have_ckpt = os.path.isdir(out_dir) and any(
        p.startswith("checkpoint-") for p in os.listdir(out_dir)
    )
    if have_ckpt:
        print(f"resuming from a checkpoint in {out_dir}")
    trainer.train(resume_from_checkpoint=have_ckpt)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir
