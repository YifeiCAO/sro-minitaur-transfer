"""Phase 0c (part 1) -- interpretable per-person features per source task.

The SRO scalar DVs already *are* the handcrafted, interpretable person-level
features the plan calls for: hDDM drift / threshold / non-decision, accuracy,
criterion/bias, lapse, condition-difference scores. For a source task A its
feature block is simply the set of DV columns named ``A.*``.

These blocks feed the handcrafted transfer matrix (the cheap "is there any
signal at all" diagnostic) and serve as the strong interpretable baseline the
LLM person-encoder must beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def task_feature_block(dvs: pd.DataFrame, task: str) -> pd.DataFrame:
    """All DV columns belonging to ``task`` (prefix ``task.``)."""
    cols = [c for c in dvs.columns if c.split(".")[0] == task]
    return dvs[cols].copy()


def all_feature_blocks(dvs: pd.DataFrame, tasks: list[str]) -> dict[str, pd.DataFrame]:
    return {t: task_feature_block(dvs, t) for t in tasks if task_feature_block(dvs, t).shape[1]}


def clean_block(block: pd.DataFrame, min_coverage: float = 0.5) -> pd.DataFrame:
    """Drop sparse columns, median-impute, z-score. Returns a numeric matrix."""
    keep = block.columns[block.notna().mean() >= min_coverage]
    b = block[keep]
    b = b.fillna(b.median())
    sd = b.std(ddof=0).replace(0, np.nan)
    z = (b - b.mean()) / sd
    return z.dropna(axis=1, how="all").fillna(0.0)
