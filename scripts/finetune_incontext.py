#!/usr/bin/env python
"""Fine-tune M_pop to USE cross-task context, on MANY (source->target) pairs.

For each training person and each within-domain ordered pair (A,B), build a
[A-session + B-session] sequence (loss only on B's responses) and mix them all.
The model learns the GENERAL skill "read a prior task's transcript, predict the
current task" -- not to memorize one target. One model then generalizes to any
pair. Standard LoRA fine-tune (no soft-prompt fragility).

    python scripts/finetune_incontext.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --subset starting_subset --pairs within \
        --out /content/drive/MyDrive/sro_minitaur/mpop_ic --max-seq-len 6144

Then eval any pair: scripts/run_incontext.py --mpop <out> --source A --target B
"""
import argparse, os, random, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.model.incontext import _b_label_mask
from sro_transfer.model.mpop import load_for_incontext_finetune
from sro_transfer.utils import load_config, load_tasks


def build_example(a_text, b_text, tok, max_len):
    a_ids = tok(a_text, add_special_tokens=True)["input_ids"]
    b_ids, b_flags = _b_label_mask(b_text, tok)
    input_ids = a_ids + b_ids
    labels = [-100] * len(a_ids) + [t if f else -100 for t, f in zip(b_ids, b_flags)]
    if len(input_ids) > max_len:                       # keep tail (B intact)
        cut = len(input_ids) - max_len
        input_ids, labels = input_ids[cut:], labels[cut:]
    if all(x == -100 for x in labels):
        return None
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--subset", default="starting_subset")
    ap.add_argument("--pairs", choices=["within", "all"], default="within")
    ap.add_argument("--out", default="/content/drive/MyDrive/sro_minitaur/mpop_ic")
    ap.add_argument("--max-seq-len", type=int, default=6144)
    ap.add_argument("--max-examples", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1, help="per-device batch; raise on A100 (4) to use the GPU")
    ap.add_argument("--grad-accum", type=int, default=8, help="lower it when raising batch to keep effective batch ~constant")
    args = ap.parse_args()
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    cfg = load_config(args.config)
    tax = load_tasks()
    tasks = tax["subsets"][args.subset]
    domain = {t: tax["tasks"][t]["domain"] for t in tasks}

    model, tok = load_for_incontext_finetune(cfg, args.mpop)
    model.config.use_cache = False

    # load all subset sessions; one consistent split over the union of subjects
    sess = {t: load_sessions(cfg["paths"]["nl_dir"], t, "complete") for t in tasks}
    universe = sorted(set().union(*[set(s) for s in sess.values()]))
    split = make_splits(universe, [], cfg["split"]["heldout_frac"], cfg["split"]["seed"])
    train = set(split.train)

    pairs = [(a, b) for a in tasks for b in tasks
             if a != b and (args.pairs == "all" or domain[a] == domain[b])]
    items = [(a, b, w) for (a, b) in pairs for w in train if w in sess[a] and w in sess[b]]
    random.Random(cfg["split"]["seed"]).shuffle(items)
    items = items[: args.max_examples]
    print(f"pairs={len(pairs)} ({args.pairs})  candidate examples capped to {len(items)}  max_len={args.max_seq_len}")

    rows = []
    for a, b, w in items:
        ex = build_example(sess[a][w], sess[b][w], tok, args.max_seq_len)
        if ex:
            rows.append(ex)
    print(f"built {len(rows)} training sequences")
    ds = Dataset.from_list(rows)

    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, bf16=bf16_ok, fp16=not bf16_ok,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="epoch", report_to=[],
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator).train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nin-context model (multi-pair) saved -> {args.out}")


if __name__ == "__main__":
    main()
