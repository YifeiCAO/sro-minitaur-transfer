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


def identification_from_candidates(own_nll, distractor_nlls, seed=0):
    """Rank metric from one person's candidate NLLs.

    own_nll: scalar (B scored under own A). distractor_nlls: list (B under others' A).
    Returns (rank_of_own among all candidates, K). Lower NLL = better; rank 1 = own
    is the single best predictor of this person's B.
    """
    cands = np.asarray([own_nll] + list(distractor_nlls), float)
    rank = int((cands < own_nll).sum()) + 1  # ties -> best case for own
    return rank, len(cands)


def exchangeability_perm(rows, seed=0, n_perm=20000):
    """Permutation null treating own-A as just another candidate A.

    rows: list of (own_nll, [distractor_nlls]). Under H0 own is exchangeable with
    the distractors. Observed stat = mean(own - mean(distractors)). Null draws a
    random candidate as pseudo-own per person. Returns (mean_diff, perm_p, top1,
    mean_rank, chance, top1_perm_p).
    """
    rng = np.random.RandomState(seed)
    diffs, ranks, Ks = [], [], []
    obs_pseudo, top1_pseudo = [], []
    for own, distr in rows:
        distr = [x for x in distr if np.isfinite(x)]
        if not np.isfinite(own) or not distr:
            continue
        cands = np.asarray([own] + distr, float)
        diffs.append(own - np.mean(distr))
        r = int((cands < own).sum()) + 1
        ranks.append(r); Ks.append(len(cands))
    diffs = np.asarray(diffs)
    obs = float(diffs.mean())
    ranks = np.asarray(ranks)
    top1 = float((ranks == 1).mean())
    mean_rank = float(ranks.mean())
    chance = float(np.mean(1.0 / np.asarray(Ks)))

    # null: pick a random candidate as pseudo-own
    perm_means, perm_top1 = [], []
    rows_f = [(o, [x for x in d if np.isfinite(x)]) for o, d in rows
              if np.isfinite(o) and any(np.isfinite(x) for x in d)]
    for _ in range(n_perm):
        ms, t1 = [], []
        for own, distr in rows_f:
            cands = np.asarray([own] + distr, float)
            j = rng.randint(len(cands))
            pseudo = cands[j]
            others = np.delete(cands, j)
            ms.append(pseudo - others.mean())
            t1.append(1 if (cands < pseudo).sum() == 0 else 0)
        perm_means.append(np.mean(ms)); perm_top1.append(np.mean(t1))
    perm_means = np.asarray(perm_means); perm_top1 = np.asarray(perm_top1)
    perm_p = max(float((perm_means <= obs).mean()), 1.0 / n_perm)
    top1_p = max(float((perm_top1 >= top1).mean()), 1.0 / n_perm)
    return {
        "mean_own_minus_distr": obs, "perm_p": perm_p,
        "id_top1": top1, "mean_rank": mean_rank, "chance_top1": chance,
        "top1_perm_p": top1_p, "n": len(ranks),
    }
