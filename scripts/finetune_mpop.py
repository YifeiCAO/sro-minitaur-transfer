#!/usr/bin/env python
"""Phase 1 -- fine-tune M_pop on the SRO population (TRAIN subjects only), no
individual information, then save the frozen floor model.

    python scripts/finetune_mpop.py --config configs/default.yaml --subset starting_subset

Requires a GPU (Colab). Loss is masked to <<...>> response tokens.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.data import available_tasks, load_sessions, make_splits
from sro_transfer.model.mpop import train_mpop
from sro_transfer.utils import load_config, load_tasks


def gather_train_sessions(cfg, tasks):
    nl_dir = cfg["paths"]["nl_dir"]
    # reference subject universe from the first task
    ref = load_sessions(nl_dir, tasks[0], "complete")
    retest_subj = list(load_sessions(nl_dir, tasks[0], "retest"))
    split = make_splits(
        list(ref), retest_subj,
        heldout_frac=cfg["split"]["heldout_frac"], seed=cfg["split"]["seed"],
        unseen_subject=cfg["split"]["unseen_subject"],
    )
    print("split:", split.summary())
    train = set(split.train)
    sessions = {}
    for t in tasks:
        s = load_sessions(nl_dir, t, "complete")
        for wid, text in s.items():
            if wid in train:
                sessions[f"{wid}::{t}"] = text     # one row per (subject, task)
    print(f"M_pop train sessions: {len(sessions)} across {len(tasks)} tasks")
    return sessions, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subset", default="starting_subset")
    ap.add_argument("--out", default="results/mpop")
    args = ap.parse_args()
    cfg = load_config(args.config)
    taxonomy = load_tasks()

    if args.subset == "all":
        tasks = available_tasks(cfg["paths"]["nl_dir"], "complete")
    else:
        tasks = taxonomy["subsets"][args.subset]
    sessions, _ = gather_train_sessions(cfg, tasks)
    train_mpop(cfg, sessions, args.out)
    print(f"M_pop saved -> {args.out}")


if __name__ == "__main__":
    main()
