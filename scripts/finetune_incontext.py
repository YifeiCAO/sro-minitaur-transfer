#!/usr/bin/env python
"""Fine-tune M_pop to USE cross-task context: train on [A-session + B-session]
sequences with loss only on B's responses. Teaches the model to read a person's
task-A transcript and predict their task-B behavior (what zero-shot couldn't do).

    python scripts/finetune_incontext.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --source directed_forgetting --target recent_probes \
        --out /content/drive/MyDrive/sro_minitaur/mpop_ic --max-seq-len 6144 --epochs 2

Then evaluate with: scripts/run_incontext.py --mpop <out>
"""
import argparse, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.model.incontext import _b_label_mask
from sro_transfer.model.mpop import load_for_incontext_finetune
from sro_transfer.utils import load_config


def build_examples(A, B, ids, tok, max_len):
    rows = []
    for w in ids:
        if w not in A or w not in B:
            continue
        a_ids = tok(A[w], add_special_tokens=True)["input_ids"]
        b_ids, b_flags = _b_label_mask(B[w], tok)
        input_ids = a_ids + b_ids
        labels = [-100] * len(a_ids) + [t if f else -100 for t, f in zip(b_ids, b_flags)]
        if len(input_ids) > max_len:                       # keep tail (B intact)
            cut = len(input_ids) - max_len
            input_ids, labels = input_ids[cut:], labels[cut:]
        if all(x == -100 for x in labels):
            continue
        rows.append({"input_ids": input_ids,
                     "attention_mask": [1] * len(input_ids), "labels": labels})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="/content/drive/MyDrive/sro_minitaur/mpop_ic")
    ap.add_argument("--max-seq-len", type=int, default=6144)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    cfg = load_config(args.config)
    model, tok = load_for_incontext_finetune(cfg, args.mpop)
    model.config.use_cache = False

    A = load_sessions(cfg["paths"]["nl_dir"], args.source, "complete")
    B = load_sessions(cfg["paths"]["nl_dir"], args.target, "complete")
    retest = list(load_sessions(cfg["paths"]["nl_dir"], args.target, "retest"))
    split = make_splits(list(B), retest, cfg["split"]["heldout_frac"], cfg["split"]["seed"])
    rows = build_examples(A, B, split.train, tok, args.max_seq_len)
    print(f"train examples = {len(rows)}  [{args.source}+{args.target}]  max_len={args.max_seq_len}")
    ds = Dataset.from_list(rows)

    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, bf16=bf16_ok, fp16=not bf16_ok,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="epoch", report_to=[],
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator).train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nin-context model saved -> {args.out}")
    print(f"evaluate: python scripts/run_incontext.py --mpop {args.out} "
          f"--source {args.source} --target {args.target} --max-seq-len {args.max_seq_len}")


if __name__ == "__main__":
    main()
