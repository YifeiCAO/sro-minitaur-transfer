#!/usr/bin/env python
"""Identification CEILING: same-task across sessions (time1 -> time2).

Using the surprise rep, can we identify a person on the SAME task from their
retest (time2) session, given their time1 session? This is the upper bound for
cross-task identification -- how identifiable a person is when NOTHING has to
transfer across tasks. Contextualizes the cross-task transfer numbers.

    python scripts/retest_ceiling.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --subset starting_subset
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sro_transfer.data import load_sessions
from sro_transfer.model.surprise import build_or_load_profiles, summarize_profile
from sro_transfer.runtime import get_model
from sro_transfer.utils import load_config, load_tasks


def _identify(V1, V2, ids, K, seed):
    rng = np.random.RandomState(seed)
    T = np.stack([V2[w] for w in ids]); T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    hits = 0
    for i, w in enumerate(ids):
        p = V1[w] / (np.linalg.norm(V1[w]) + 1e-8)
        others = [j for j in range(len(ids)) if j != i]
        cand = [i] + list(rng.choice(others, size=min(K - 1, len(others)), replace=False))
        if cand[int(np.argmax(T[cand] @ p))] == i:
            hits += 1
    return hits / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--subset", default="starting_subset")
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed, max_len, rdir = cfg["split"]["seed"], cfg["model"]["max_seq_len"], cfg["paths"]["results"]
    tasks = load_tasks()["subsets"][args.subset]
    model, tok = get_model(cfg, args.mpop)

    rows = []
    for t in tasks:
        s1 = load_sessions(cfg["paths"]["nl_dir"], t, "complete")
        s2 = load_sessions(cfg["paths"]["nl_dir"], t, "retest")
        P1 = build_or_load_profiles(model, tok, s1, Path(rdir) / "surprise" / f"{t}.pt", max_len)
        P2 = build_or_load_profiles(model, tok, s2, Path(rdir) / "surprise_retest" / f"{t}.pt", max_len)
        common = [w for w in (set(P1) & set(P2))]
        if len(common) < 20:
            print(f"  {t}: only {len(common)} retest subjects, skipping"); continue
        V1 = {w: summarize_profile(P1[w]) for w in common}
        V2 = {w: summarize_profile(P2[w]) for w in common}
        # standardize with time1 stats
        M = np.stack([V1[w] for w in common]); mu, sd = M.mean(0), M.std(0) + 1e-6
        V1 = {w: (V1[w] - mu) / sd for w in common}; V2 = {w: (V2[w] - mu) / sd for w in common}
        top1 = _identify(V1, V2, common, args.K, seed)
        rows.append({"task": t, "n": len(common), "ceiling_top1": round(top1, 3)})
        print(f"  {t:<28} ceiling top1 = {top1:.3f}  (n={len(common)})")

    mean = float(np.mean([r["ceiling_top1"] for r in rows])) if rows else float("nan")
    out = {"phase": "retest-ceiling", "K": args.K, "chance": 1.0 / args.K,
           "mean_ceiling_top1": mean, "per_task": rows}
    Path(rdir).mkdir(parents=True, exist_ok=True)
    (Path(rdir) / "retest_ceiling.json").write_text(json.dumps(out, indent=2))
    print(f"\nmean same-task retest identification (CEILING) = {mean:.3f}  (chance {1/args.K:.3f})")
    print("compare cross-task transfer to this: transfer/ceiling = fraction of identifiable signal that crosses tasks")


if __name__ == "__main__":
    main()
