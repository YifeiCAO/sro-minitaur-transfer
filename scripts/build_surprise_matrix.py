#!/usr/bin/env python
"""Cross-task transfer MATRIX from surprise reps (the pivot away from soft-prompt).

Surprise profiles are extracted ONCE per task; every pairwise transfer is then a
cheap linear map + held-out identification. Builds T_surprise[A,B] (top1 above
chance) over a task subset, plus hub ranking and within/across-domain means --
the LLM-rep analogue of the handcrafted 0c matrix.

    python scripts/build_surprise_matrix.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --subset starting_subset
First run extracts + caches each task's profiles (~10 min/task); then instant.
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.model.surprise import build_or_load_profiles, summarize_profile
from sro_transfer.runtime import get_model
from sro_transfer.utils import load_config, load_tasks


def _identify(pred, true_vecs, ids, K, seed):
    rng = np.random.RandomState(seed)
    T = np.stack([true_vecs[w] for w in ids])
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    hits = 0
    for i in range(len(ids)):
        p = pred[i] / (np.linalg.norm(pred[i]) + 1e-8)
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
    from sklearn.linear_model import Ridge

    cfg = load_config(args.config)
    seed, max_len = cfg["split"]["seed"], cfg["model"]["max_seq_len"]
    rdir = cfg["paths"]["results"]
    tax = load_tasks()
    tasks = sorted(tax["tasks"]) if args.subset == "all" else tax["subsets"][args.subset]
    domain = {t: tax["tasks"][t]["domain"] for t in tasks}
    model, tok = get_model(cfg, args.mpop)

    # one-time: per-task per-person summary vectors + a split
    summ, splits = {}, {}
    for t in tasks:
        sess = load_sessions(cfg["paths"]["nl_dir"], t, "complete")
        prof = build_or_load_profiles(model, tok, sess, Path(rdir) / "surprise" / f"{t}.pt", max_len)
        summ[t] = {w: summarize_profile(p) for w, p in prof.items()}
        retest = list(load_sessions(cfg["paths"]["nl_dir"], t, "retest"))
        splits[t] = make_splits(list(sess), retest, cfg["split"]["heldout_frac"], seed)
        print(f"  profiled {t}: {len(summ[t])} subjects")

    T = pd.DataFrame(index=tasks, columns=tasks, dtype=float)
    for a in tasks:
        for b in tasks:
            if a == b:
                continue
            common = set(summ[a]) & set(summ[b])
            tr = [w for w in splits[b].train if w in common]
            he = [w for w in splits[b].heldout if w in common]
            if len(tr) < 30 or len(he) < 20:
                continue
            Xtr = np.stack([summ[a][w] for w in tr]); Ytr = np.stack([summ[b][w] for w in tr])
            Xte = np.stack([summ[a][w] for w in he])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-6
            pred = Ridge(alpha=10.0).fit((Xtr - mu) / sd, (Ytr - ym) / ys).predict((Xte - mu) / sd)
            Tn = {w: (summ[b][w] - ym) / ys for w in he}
            T.loc[a, b] = _identify(pred, Tn, he, args.K, seed)

    chance = 1.0 / args.K
    hubs = pd.DataFrame({
        "as_source": T.mean(axis=1, skipna=True),
        "as_target": T.mean(axis=0, skipna=True),
    })
    hubs["hub"] = hubs.mean(axis=1)
    within = [T.loc[a, b] for a in tasks for b in tasks
              if a != b and not pd.isna(T.loc[a, b]) and domain[a] == domain[b]]
    across = [T.loc[a, b] for a in tasks for b in tasks
              if a != b and not pd.isna(T.loc[a, b]) and domain[a] != domain[b]]

    out = Path(rdir)
    T.to_csv(out / "surprise_matrix.csv")
    hubs.sort_values("hub", ascending=False).to_csv(out / "surprise_hubs.csv")
    print("\n=== T_surprise[A,B] identification top1 (chance =", chance, ") ===")
    print(T.round(3).to_string())
    print("\nhubs:\n", hubs.sort_values("hub", ascending=False).round(3).to_string())
    print(f"\nwithin-domain mean top1 = {np.nanmean(within):.3f}  (n={len(within)})")
    print(f"across-domain mean top1 = {np.nanmean(across):.3f}  (n={len(across)})")
    print(f"chance = {chance:.3f}")
    json.dump({"within": float(np.nanmean(within)), "across": float(np.nanmean(across)),
               "chance": chance}, open(out / "surprise_matrix_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
