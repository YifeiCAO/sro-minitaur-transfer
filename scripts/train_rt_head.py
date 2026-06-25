#!/usr/bin/env python
"""Train Approach B: continuous log-RT regression head + LoRA, jointly with choice.

    python scripts/train_rt_head.py --subset all \
        --nl-dir /content/drive/MyDrive/sro_minitaur/output_nl_rtval \
        --out /content/drive/MyDrive/sro_minitaur/mpop_rt_head

Needs output_nl_rtval (choice-only text + rt_values sidecar). A100-40G: batch 4
(no liger here -- the RT head hooks the lm_head input, so logits aren't fused).
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.data import make_splits
from sro_transfer.model.rt_head import train_rt_head
from sro_transfer.utils import load_config, load_tasks


def load_rtval(nl_dir, task, source="complete"):
    fp = Path(nl_dir) / source / f"{task}.all.jsonl"
    out = {}
    if fp.exists():
        for line in open(fp, encoding="utf-8"):
            o = json.loads(line)
            out[o["worker_id"]] = (o["text"], o.get("rt_values", []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--nl-dir", default="/content/drive/MyDrive/sro_minitaur/output_nl_rtval")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--out", default="/content/drive/MyDrive/sro_minitaur/mpop_rt_head")
    ap.add_argument("--lam", type=float, default=1.0, help="weight on the RT NLL vs choice CE")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke test); 0 = full")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["paths"]["nl_dir"] = args.nl_dir
    tax = load_tasks()
    tasks = sorted(tax["tasks"]) if args.subset == "all" else tax["subsets"][args.subset]

    per_task = {t: load_rtval(args.nl_dir, t) for t in tasks}
    per_task = {t: s for t, s in per_task.items() if s}
    universe = sorted(set().union(*[set(s) for s in per_task.values()]))
    split = make_splits(universe, [], cfg["split"]["heldout_frac"], cfg["split"]["seed"])
    train = set(split.train)
    print(f"split: train={len(split.train)} heldout={len(split.heldout)}")

    sessions = {}
    for t, s in per_task.items():
        for wid, (text, rtv) in s.items():
            if wid in train:
                sessions[f"{wid}::{t}"] = (text, rtv)
    print(f"RT-head train sessions: {len(sessions)} across {len(per_task)} tasks")

    train_rt_head(cfg, sessions, args.out, lam=args.lam, epochs=args.epochs,
                  batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
                  save_steps=args.save_steps, max_len=args.max_seq_len, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
