"""Phase 3 -- cross-task identification (the PRIMARY result).

Given person p's source-task embedding z_p, pick p out of K candidate *target*
sessions (p's real one + K-1 other people's), choosing the candidate whose
responses z_p makes least surprising. Rank-based metrics dodge the NLL dilution
problem: even a small per-token benefit shows up cleanly as above-chance
identification. Chance = 1/K.

The upper bound is same-task retest identification (source == target, time1 z vs
time2 sessions): how identifiable a person is when nothing has to transfer.

Pure orchestration over a ``score_fn(text, z) -> nll`` and a ``z_of(wid) -> z``;
works with the floor scorer (z ignored -> chance, a sanity check) or the
transfer scorer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def identify(
    target_sessions: dict[str, str],
    z_of,
    score_fn,
    K: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    wids = list(target_sessions)
    rng = np.random.RandomState(seed)
    rows = []
    for p in wids:
        z = z_of(p)
        others = [w for w in wids if w != p]
        if not others:
            continue
        k = min(K - 1, len(others))
        distract = list(rng.choice(others, size=k, replace=False))
        cands = [p] + distract
        nlls = np.array([score_fn(target_sessions[w], z) for w in cands])
        order = np.argsort(nlls, kind="stable")           # ascending NLL
        ranked = [cands[i] for i in order]
        rank = ranked.index(p) + 1
        rows.append({"wid": p, "rank": rank, "K": len(cands), "top1": rank == 1})
    return pd.DataFrame(rows)


def identification_report(res: pd.DataFrame) -> dict:
    if res.empty:
        return {}
    K = float(res["K"].mean())
    return {
        "n": int(len(res)),
        "K": K,
        "chance_top1": 1.0 / K,
        "top1_acc": float(res["top1"].mean()),
        "mean_rank": float(res["rank"].mean()),
        "mrr": float((1.0 / res["rank"]).mean()),
        # normalized rank in [0,1]; 0.5 = chance, ->0 = perfect
        "norm_rank": float(((res["rank"] - 1) / (res["K"] - 1)).mean()),
    }
