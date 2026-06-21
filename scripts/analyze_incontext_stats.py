#!/usr/bin/env python
"""OFFLINE rigorous stats for the in-context matrix (no GPU).

Reads the raw per-person candidate NLLs saved by build_incontext_matrix.py
(results/incontext_matrix/raw/*.json) and produces, per pair, the honest bundle
the audit demanded:
  - sign-flip permutation p (primary; replaces the inflated t-test)
  - n_shuffle=1 contrast (real vs a single distractor; no variance averaging)
  - exchangeability permutation (own-A as just another candidate) + K-identification
    top-1 / mean-rank with its own permutation p
  - Cohen's dz, bootstrap-over-people 95% CI, Wilcoxon, sign test
Then across pairs: Benjamini-Hochberg FDR on the permutation p's, and the
within- vs across-domain identification contrast.

    python scripts/analyze_incontext_stats.py            # reads default results dir
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sro_transfer.stats import exchangeability_perm, paired_report
from sro_transfer.utils import load_config, load_tasks


def bh_fdr(pvals):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dir", default=None, help="raw dir (default: <results>/incontext_matrix/raw)")
    ap.add_argument("--subset", default="starting_subset")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rawdir = Path(args.dir) if args.dir else Path(cfg["paths"]["results"]) / "incontext_matrix" / "raw"
    tax = load_tasks()
    domain = {t: tax["tasks"][t]["domain"] for t in tax["tasks"]}

    files = sorted(rawdir.glob("*.json"))
    if not files:
        print(f"no raw files in {rawdir} — run build_incontext_matrix.py first"); return

    results = []
    for fp in files:
        rows = json.loads(fp.read_text())
        a, b = fp.stem.split("__")
        real = [r["real"] for r in rows]
        shuf_mean = [float(np.mean(list(r["distractors"].values()))) for r in rows]
        shuf_one = [list(r["distractors"].values())[0] for r in rows]
        ex_rows = [(r["real"], list(r["distractors"].values())) for r in rows]
        ex = exchangeability_perm(ex_rows, seed=cfg["split"]["seed"])
        rep_mean = paired_report(real, shuf_mean, seed=cfg["split"]["seed"])
        rep_one = paired_report(real, shuf_one, seed=cfg["split"]["seed"])
        results.append({
            "source": a, "target": b, "within": domain.get(a) == domain.get(b),
            "n": ex["n"], "n_degenerate": ex.get("n_degenerate", 0),
            "id_top1": ex["id_top1"], "chance_top1": ex["chance_top1"],
            "mean_rank": ex["mean_rank"], "id_top1_perm_p": ex["top1_perm_p"],
            "mean_diff": rep_mean["mean_diff"], "cohen_dz": rep_mean["cohen_dz"],
            "perm_p": rep_mean["perm_p"], "boot_ci95": rep_mean["boot_ci95"],
            "perm_p_nshuffle1": rep_one["perm_p"], "mean_diff_nshuffle1": rep_one["mean_diff"],
            "wilcoxon_p": rep_mean["wilcoxon_p"], "frac_below": rep_mean["frac_real_below"],
        })

    qs = bh_fdr([r["perm_p"] for r in results])
    for r, q in zip(results, qs):
        r["perm_q_BH"] = float(q)

    results.sort(key=lambda r: (np.isnan(r["perm_p"]), r["perm_p"]))
    print(f"{'pair':<48} {'win':<4} {'id_top1':>8} {'rank':>6} {'mdiff':>8} {'dz':>6} "
          f"{'perm_p':>9} {'q_BH':>8} {'degen':>6}")
    for r in results:
        print(f"{r['source']+'>'+r['target']:<48} {'Y' if r['within'] else 'n':<4} "
              f"{r['id_top1']:>8.3f} {r['mean_rank']:>6.2f} {r['mean_diff']:>+8.4f} "
              f"{r['cohen_dz']:>6.2f} {r['perm_p']:>9.1e} {r['perm_q_BH']:>8.1e} {r['n_degenerate']:>6d}")

    # exclude degenerate (context-truncated) pairs from the within/across contrast
    valid = [r for r in results if np.isfinite(r["id_top1"]) and r["n"] >= 20]
    within = [r for r in valid if r["within"]]
    across = [r for r in valid if not r["within"]]
    n_sig = sum(1 for r in valid if r["perm_q_BH"] < 0.05)
    summ = {
        "n_pairs": len(results), "n_pairs_valid": len(valid), "n_sig_BH_q<.05": n_sig,
        "within_id_top1_mean": float(np.mean([r["id_top1"] for r in within])) if within else None,
        "across_id_top1_mean": float(np.mean([r["id_top1"] for r in across])) if across else None,
        "chance_top1": valid[0]["chance_top1"] if valid else None,
        "within_mean_diff": float(np.mean([r["mean_diff"] for r in within])) if within else None,
        "across_mean_diff": float(np.mean([r["mean_diff"] for r in across])) if across else None,
        "dropped_degenerate_pairs": [f"{r['source']}>{r['target']}" for r in results if r not in valid],
    }
    out = rawdir.parent / "incontext_stats.json"
    out.write_text(json.dumps({"summary": summ, "pairs": results}, indent=2))
    print("\nsummary:", json.dumps(summ, indent=2))
    print(f"\n{n_sig}/{len(results)} pairs significant at BH q<.05")
    if within and across:
        print(f"within id_top1 {summ['within_id_top1_mean']:.3f} vs across "
              f"{summ['across_id_top1_mean']:.3f} (chance {summ['chance_top1']:.3f})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
