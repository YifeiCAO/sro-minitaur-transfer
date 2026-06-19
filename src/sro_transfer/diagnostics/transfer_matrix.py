"""Phase 0c (part 2) -- the handcrafted directed transfer matrix T[A, B].

T[A, B] = how well person-level features from source task A predict target task
B's behavior, above a population baseline, cross-validated and reliability-
normalized. This is the CHEAP diagnostic that gates the expensive 8B work:

  * whole matrix ~ 0  -> task-based individual differences barely transfer;
                         shrink scope before building the model rig.
  * structure present -> note the hubs and strong (A,B) pairs; they become the
                         targets and the strong baseline for the LLM stage.

This fast version predicts B's *scalar DV block* from A's features (ridge, k-fold
CV). The trial-level version of the same question is Phase 2 (the LLM person-
encoder); T_handcrafted is what that model must beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

from .handcrafted import clean_block, task_feature_block


def _aligned(dvs: pd.DataFrame, a: str, b: str):
    XA = clean_block(task_feature_block(dvs, a))
    YB = clean_block(task_feature_block(dvs, b))
    idx = XA.index.intersection(YB.index)
    XA, YB = XA.loc[idx], YB.loc[idx]
    # require enough subjects and non-empty blocks
    if XA.shape[1] == 0 or YB.shape[1] == 0 or len(idx) < 50:
        return None
    return XA.to_numpy(), YB.to_numpy()


def transfer_score(
    dvs: pd.DataFrame, a: str, b: str, reliability_b: float | None = None,
    n_splits: int = 5, seed: int = 0,
) -> dict:
    """Cross-validated predictability of B's DV block from A's features."""
    al = _aligned(dvs, a, b)
    if al is None:
        return {"raw": np.nan, "norm": np.nan, "n": 0}
    X, Y = al
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = RidgeCV(alphas=np.logspace(-2, 4, 13))
    # per target column CV correlation between prediction and truth
    cors = []
    for j in range(Y.shape[1]):
        yj = Y[:, j]
        if np.std(yj) == 0:
            continue
        pred = cross_val_predict(model, X, yj, cv=kf)
        if np.std(pred) == 0:
            continue
        cors.append(np.corrcoef(pred, yj)[0, 1])
    if not cors:
        return {"raw": np.nan, "norm": np.nan, "n": X.shape[0]}
    raw = float(np.nanmean(cors))
    norm = raw
    if reliability_b and reliability_b > 0.05:
        norm = float(np.clip(raw / np.sqrt(reliability_b), -1.5, 1.5))
    return {"raw": raw, "norm": norm, "n": X.shape[0]}


def transfer_matrix(
    dvs: pd.DataFrame, tasks: list[str],
    reliability: dict[str, float] | None = None, value: str = "norm",
) -> pd.DataFrame:
    """Directed T[source, target]; diagonal is NaN (same-task is reliability)."""
    reliability = reliability or {}
    T = pd.DataFrame(index=tasks, columns=tasks, dtype=float)
    for a in tasks:
        for b in tasks:
            if a == b:
                continue
            T.loc[a, b] = transfer_score(dvs, a, b, reliability.get(b))[value]
    return T


def hub_ranking(T: pd.DataFrame) -> pd.DataFrame:
    """Rank tasks by mean outgoing (source) and incoming (target) transfer."""
    out = pd.DataFrame(
        {
            "as_source_mean": T.mean(axis=1, skipna=True),   # row mean
            "as_target_mean": T.mean(axis=0, skipna=True),   # col mean
        }
    )
    out["hub_score"] = out.mean(axis=1)
    return out.sort_values("hub_score", ascending=False)


def domain_structure(T: pd.DataFrame, domain: dict[str, str]) -> dict:
    """Within- vs across-domain mean transfer (does T recover task structure?)."""
    within, across = [], []
    for a in T.index:
        for b in T.columns:
            if a == b or pd.isna(T.loc[a, b]):
                continue
            (within if domain.get(a) == domain.get(b) else across).append(T.loc[a, b])
    return {
        "within_domain_mean": float(np.mean(within)) if within else np.nan,
        "across_domain_mean": float(np.mean(across)) if across else np.nan,
        "n_within": len(within),
        "n_across": len(across),
    }
