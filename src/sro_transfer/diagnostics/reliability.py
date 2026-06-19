"""Phase 0b -- test-retest reliability ceiling.

The retest subjects did every task twice. The reliability of a dependent
variable is the agreement between time1 (``complete``) and time2 (``retest``)
over those subjects. No model can predict more of a measure than its reliable
(true-score) variance, so this is the ceiling against which every transfer
number is normalized.

Outputs a per-DV table (Pearson / Spearman / ICC) and a per-task aggregate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_dvs(path: str | Path) -> pd.DataFrame:
    """Load a SRO scalar-DV CSV (subjects x DVs), indexed by worker id."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    # keep only numeric DV columns
    return df.select_dtypes(include=[np.number])


def icc2(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(2,1): two-way random, absolute agreement, single measurement."""
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape  # k == 2 raters
    if n < 3:
        return np.nan
    grand = m.mean()
    row = m.mean(axis=1)
    col = m.mean(axis=0)
    ss_total = ((m - grand) ** 2).sum()
    ss_row = k * ((row - grand) ** 2).sum()
    ss_col = n * ((col - grand) ** 2).sum()
    ss_err = ss_total - ss_row - ss_col
    msr = ss_row / (n - 1)
    msc = ss_col / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom != 0 else np.nan


def test_retest(complete: pd.DataFrame, retest: pd.DataFrame) -> pd.DataFrame:
    """Per-DV reliability over subjects present in both sessions."""
    shared_subj = complete.index.intersection(retest.index)
    shared_dv = complete.columns.intersection(retest.columns)
    rows = []
    for dv in shared_dv:
        a = complete.loc[shared_subj, dv]
        b = retest.loc[shared_subj, dv]
        ok = a.notna() & b.notna()
        n = int(ok.sum())
        if n < 5 or a[ok].std() == 0 or b[ok].std() == 0:
            continue
        av, bv = a[ok].to_numpy(), b[ok].to_numpy()
        rows.append(
            {
                "dv": dv,
                "task": dv.split(".")[0],
                "n": n,
                "pearson": float(stats.pearsonr(av, bv)[0]),
                "spearman": float(stats.spearmanr(av, bv)[0]),
                "icc": icc2(av, bv),
            }
        )
    return pd.DataFrame(rows).sort_values("icc", ascending=False).reset_index(drop=True)


def per_task_reliability(rel: pd.DataFrame, metric: str = "icc") -> pd.DataFrame:
    """Aggregate DV reliabilities to one number per task (mean + best)."""
    g = rel.groupby("task")[metric]
    out = pd.DataFrame(
        {
            "n_dv": g.size(),
            f"mean_{metric}": g.mean(),
            f"max_{metric}": g.max(),
        }
    )
    return out.sort_values(f"mean_{metric}", ascending=False)
