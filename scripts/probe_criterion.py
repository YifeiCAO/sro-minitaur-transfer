#!/usr/bin/env python
"""CRITERION PROBE: does the FM's per-person SURPRISE rep predict an OUT-OF-MODEL
SRO measure (a survey score the model never saw)? Cross-validated.

This turns "an artifact transfers across tasks" into "the transferable individual
signal is psychologically MEANINGFUL" — the single biggest lever on impact.
No GPU: works off the cached surprise profiles + the SRO DV/survey CSV.

    # 1) see candidate survey targets in your DV csv:
    python scripts/probe_criterion.py --tag 8b_raw --list
    # 2) probe chosen targets (comma-separated column names):
    python scripts/probe_criterion.py --tag 8b_raw \
        --targets upps_impulsivity_survey.sensation_seeking,bis11_survey.Nonplanning
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

from sro_transfer.model.surprise import summarize_profile
from sro_transfer.utils import load_config, load_tasks


def person_reps(rdir, tag, tasks):
    """Concatenated per-person surprise summary across tasks (subjects common to all)."""
    import torch
    tagsuf = f"_{tag}" if tag else ""
    per = {}
    for t in tasks:
        fp = Path(rdir) / f"surprise{tagsuf}" / f"{t}.pt"
        if fp.exists():
            per[t] = {w: summarize_profile(p) for w, p in torch.load(fp).items()}
    if not per:
        return {}, []
    loaded = sorted(per)
    common = set.intersection(*[set(per[t]) for t in loaded])
    reps = {w: np.concatenate([per[t][w] for t in loaded]) for w in common}
    return reps, loaded


def _cv_r(X, y, cv, seed):
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-1, 3, 9)), X, y, cv=cv)
    return float(stats.pearsonr(y, pred)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="8b_raw", help="which surprise cache (e.g. 8b_raw, 70b_raw, '' for mpop)")
    ap.add_argument("--subset", default="starting_subset")
    ap.add_argument("--targets", default="", help="comma-separated DV/survey column names")
    ap.add_argument("--list", action="store_true", help="list survey-like columns and exit")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dv = pd.read_csv(cfg["paths"]["dv_complete"], index_col=0)
    if args.list:
        cols = [c for c in dv.columns if "survey" in c.lower()]
        print(f"{len(cols)} survey-like columns in {cfg['paths']['dv_complete']}:")
        for c in cols:
            print("  ", c)
        print("\n(no 'survey' match? run without --list logic: print(dv.columns.tolist()))")
        return

    tasks = load_tasks()["subsets"][args.subset]
    reps, loaded = person_reps(cfg["paths"]["results"], args.tag, tasks)
    if not reps:
        print(f"no cached surprise profiles for tag '{args.tag}' — build them first"); return
    wids = [w for w in reps if w in dv.index]
    X = np.stack([reps[w] for w in wids])
    print(f"rep: tag='{args.tag}'  tasks={len(loaded)}  dim={X.shape[1]}  n_subj={len(wids)}\n")

    rng = np.random.RandomState(cfg["split"]["seed"])
    cv = KFold(args.folds, shuffle=True, random_state=cfg["split"]["seed"])
    rows = []
    for tgt in [t for t in args.targets.split(",") if t]:
        if tgt not in dv.columns:
            print(f"  [missing column] {tgt}"); continue
        y = pd.to_numeric(dv.loc[wids, tgt], errors="coerce").values
        ok = np.isfinite(y)
        Xo, yo = X[ok], y[ok]
        if len(yo) < 50:
            print(f"  [too few n={len(yo)}] {tgt}"); continue
        r = _cv_r(Xo, yo, cv, cfg["split"]["seed"])
        null = np.array([_cv_r(Xo, rng.permutation(yo), cv, 0) for _ in range(args.n_perm)])
        p = (np.sum(null >= r) + 1) / (args.n_perm + 1)
        rows.append({"target": tgt, "n": int(len(yo)), "cv_r": round(r, 3), "perm_p": round(float(p), 4)})
        print(f"  {tgt:52s}  n={len(yo):3d}  CV r={r:+.3f}  perm_p={p:.3f}  {'*' if p < .05 else ''}")

    if rows:
        out = Path(cfg["paths"]["results"]) / f"criterion_probe_{args.tag or 'mpop'}.json"
        out.write_text(json.dumps(rows, indent=2))
        sig = [r for r in rows if r["perm_p"] < 0.05]
        print(f"\n{len(sig)}/{len(rows)} targets predicted above chance (perm_p<.05)")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
