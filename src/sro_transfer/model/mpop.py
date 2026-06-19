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
    """Load the behavioral FM (Minitaur/Centaur) + tokenizer, 4-bit for Colab.

    The base is ``base_llm`` with the ``base_model`` LoRA adapter applied and
    merged; a *fresh* LoRA is then attached for the SRO population fine-tune so
    the original Psych-101 adapter is preserved as the starting point.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["base_llm"])
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
        m["base_llm"], quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # apply the existing behavioral-FM adapter (Minitaur / Centaur), then merge
    if m.get("base_model"):
        model = PeftModel.from_pretrained(model, m["base_model"])
        model = model.merge_and_unload()

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


def build_dataset(sessions: dict[str, str], tokenizer, max_len: int):
    """HF Dataset of response-masked sessions (one row per (subject, task))."""
    from datasets import Dataset

    rows = [{"text": t} for t in sessions.values()]
    ds = Dataset.from_list(rows)

    def _map(ex):
        return build_labels(ex["text"], tokenizer, max_len)

    return ds.map(_map, remove_columns=["text"])


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
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir
