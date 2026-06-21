"""Honest paired statistics for the in-context transfer result.

The audit flagged that ``ttest_rel`` on a variance-deflated, non-independent
shuffled arm overstates significance (p=7e-20). This module replaces it with a
distribution-free bundle that makes no normality / independence-across-people
assumption: a sign-flip permutation null is the primary p-value.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _ss


def paired_report(real, comp, n_perm=20000, seed=0):
    """Compare paired per-person NLLs ``real`` vs ``comp`` (e.g. shuffled/matched).

    Negative mean_diff => real is LOWER (own-A helps). Primary p = sign-flip
    permutation (exact under per-person exchangeability), which sidesteps the
    inflated t-test. Also returns Cohen's dz, bootstrap-over-people 95% CI,
    Wilcoxon, sign test, and a binomial CI on the fraction real<comp.
    """
    real = np.asarray(real, float)
    comp = np.asarray(comp, float)
    ok = np.isfinite(real) & np.isfinite(comp)
    real, comp = real[ok], comp[ok]
    d = real - comp
    n = len(d)
    rng = np.random.RandomState(seed)

    obs = float(d.mean())
    dz = obs / (d.std(ddof=1) + 1e-12)

    # sign-flip permutation null
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * d).mean(axis=1)
    perm_p = float((np.abs(perm_means) >= abs(obs)).mean())
    perm_p = max(perm_p, 1.0 / n_perm)

    # bootstrap-over-people 95% CI on the mean diff
    boot = np.array([d[rng.randint(0, n, n)].mean() for _ in range(n_perm // 2)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    below = int((real < comp).sum())
    # exact binomial CI (Clopper-Pearson) on frac below
    lo = _ss.beta.ppf(0.025, below, n - below + 1) if below > 0 else 0.0
    hi = _ss.beta.ppf(0.975, below + 1, n - below) if below < n else 1.0

    try:
        wil = float(_ss.wilcoxon(real, comp).pvalue)
    except ValueError:
        wil = float("nan")
    sign_p = float(_ss.binomtest(below, n, 0.5).pvalue)
    t_p = float(_ss.ttest_rel(real, comp).pvalue)

    return {
        "n": n,
        "mean_diff": obs,
        "cohen_dz": float(dz),
        "perm_p": perm_p,
        "boot_ci95": [ci[0], ci[1]],
        "frac_real_below": below / n,
        "frac_ci95": [float(lo), float(hi)],
        "wilcoxon_p": wil,
        "sign_p": sign_p,
        "ttest_p_DEPRECATED": t_p,
    }


def _fair_rank(own, cands):
    """Average-rank of `own` among `cands` with FAIR tie handling.

    Ties get the mean rank (so an all-equal row -> ~middle rank ~= chance, not 1).
    Returns (rank, is_unique_top1, degenerate) where degenerate=True means the
    candidates carry ~no spread (e.g. context was truncated away -> all equal).
    """
    cands = np.asarray(cands, float)
    spread = float(np.nanmax(cands) - np.nanmin(cands)) if len(cands) else 0.0
    n_less = int((cands < own).sum())
    n_eq = int((cands == own).sum())
    rank = n_less + (n_eq + 1) / 2.0          # average rank over the tie block
    top1 = (n_less == 0) and (n_eq == 1)       # own must be the UNIQUE minimum
    return rank, top1, spread < 1e-9


def identification_from_candidates(own_nll, distractor_nlls, seed=0):
    """Rank metric from one person's candidate NLLs (fair ties)."""
    cands = [own_nll] + list(distractor_nlls)
    rank, _, _ = _fair_rank(own_nll, cands)
    return rank, len(cands)


def exchangeability_perm(rows, seed=0, n_perm=20000):
    """Permutation null treating own-A as just another candidate A.

    rows: list of (own_nll, [distractor_nlls]). Under H0 own is exchangeable with
    the distractors. Observed stat = mean(own - mean(distractors)). Null draws a
    random candidate as pseudo-own per person. Returns (mean_diff, perm_p, top1,
    mean_rank, chance, top1_perm_p).
    """
    rng = np.random.RandomState(seed)
    # build clean candidate arrays once; drop degenerate (no-spread) rows
    clean, ranks, top1s, diffs, Ks = [], [], [], [], []
    n_degenerate = 0
    for own, distr in rows:
        distr = [x for x in distr if np.isfinite(x)]
        if not np.isfinite(own) or not distr:
            continue
        cands = np.asarray([own] + distr, float)
        rank, is_top1, degen = _fair_rank(own, cands)
        if degen:                      # context had ~no effect (e.g. A truncated away)
            n_degenerate += 1
            continue
        clean.append(cands)
        ranks.append(rank); top1s.append(1 if is_top1 else 0)
        diffs.append(own - cands[1:].mean()); Ks.append(len(cands))
    if not clean:
        return {"mean_own_minus_distr": float("nan"), "perm_p": float("nan"),
                "id_top1": float("nan"), "mean_rank": float("nan"),
                "chance_top1": float("nan"), "top1_perm_p": float("nan"),
                "n": 0, "n_degenerate": n_degenerate}
    obs = float(np.mean(diffs))
    top1 = float(np.mean(top1s))
    mean_rank = float(np.mean(ranks))
    chance = float(np.mean(1.0 / np.asarray(Ks)))

    # null: each person's own-A is just a random candidate. Per row precompute the
    # diff and unique-min for every candidate choice, then sample (vectorized).
    pm = np.zeros((n_perm, len(clean))); pt = np.zeros((n_perm, len(clean)))
    for c, cands in enumerate(clean):
        K = len(cands); s = cands.sum()
        dvals = cands * K / (K - 1) - s / (K - 1)          # pseudo - mean(others)
        t1vals = np.array([(cands < v).sum() == 0 and (cands == v).sum() == 1
                           for v in cands], float)
        idx = rng.randint(K, size=n_perm)
        pm[:, c] = dvals[idx]; pt[:, c] = t1vals[idx]
    perm_means = pm.mean(axis=1); perm_top1 = pt.mean(axis=1)
    perm_p = max(float((perm_means <= obs).mean()), 1.0 / n_perm)
    top1_p = max(float((perm_top1 >= top1).mean()), 1.0 / n_perm)
    return {
        "mean_own_minus_distr": obs, "perm_p": perm_p,
        "id_top1": top1, "mean_rank": mean_rank, "chance_top1": chance,
        "top1_perm_p": top1_p, "n": len(clean), "n_degenerate": n_degenerate,
    }
