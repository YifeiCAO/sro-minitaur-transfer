"""Phase 4 -- the headline NLL contrast: real-z vs floor vs shuffled-z.

For each held-out person p on target task B:
  floor    = NLL of p's responses under M_pop (no individual info)
  real     = NLL under M_pop + z_p (p's own source embedding)
  shuffled = NLL under M_pop + z_q (a random OTHER person's embedding)

The claim rides on the CONTRAST, not the absolute NLL:
  * real < shuffled  => the benefit is person-specific (the clean transfer signal)
  * real < floor     => z carries predictive individual information at all
Reporting real vs shuffled makes the result immune to whether M_pop is perfectly
calibrated, since both add identical capacity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def nll_floor_real_shuffled(
    target_sessions: dict[str, str],
    z_of,
    score_floor,
    score_transfer,
    n_shuffle: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-person floor / real-z / shuffled-z mean response-NLL."""
    wids = list(target_sessions)
    rng = np.random.RandomState(seed)
    rows = []
    for p in wids:
        text = target_sessions[p]
        floor = score_floor(text, None)
        real = score_transfer(text, z_of(p))
        others = [w for w in wids if w != p]
        sh = [
            score_transfer(text, z_of(q))
            for q in rng.choice(others, size=min(n_shuffle, len(others)), replace=False)
        ] if others else []
        rows.append(
            {
                "wid": p,
                "floor": floor,
                "real": real,
                "shuffled": float(np.mean(sh)) if sh else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["real", "shuffled", "floor"])
    if d.empty:
        return {}
    real_vs_shuf = d["real"] - d["shuffled"]      # want < 0
    real_vs_floor = d["real"] - d["floor"]        # want < 0
    t1 = stats.ttest_rel(d["real"], d["shuffled"])
    t2 = stats.ttest_rel(d["real"], d["floor"])
    return {
        "n": int(len(d)),
        "mean_real_minus_shuffled": float(real_vs_shuf.mean()),
        "p_real_vs_shuffled": float(t1.pvalue),
        "mean_real_minus_floor": float(real_vs_floor.mean()),
        "p_real_vs_floor": float(t2.pvalue),
        "frac_persons_real_below_shuffled": float((real_vs_shuf < 0).mean()),
    }
