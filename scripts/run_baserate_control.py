#!/usr/bin/env python
"""BASE-RATE CONTROL: is the in-context effect person-specific TRANSFER, or just
the person's marginal choice rate (base-rate matching)?

The audit's biggest risk: A and B share response vocabulary (smaller_sooner/
larger_later; yes/no), so own-A could lower own-B NLL merely by revealing the
person's marginal choice rate. To rule this out, compare own-A against a
MARGINAL-MATCHED stranger: the held-out person whose A-task response distribution
is CLOSEST to p's. If real < matched-shuffled, own-A carries person-specific
structure BEYOND the base rate.

Single distractor per arm (n_shuffle=1) -> no variance-deflation; primary p is a
sign-flip permutation. Leads with directed_forgetting->recent_probes (balanced
target, base-rate-robust).

    python scripts/run_baserate_control.py --mpop /content/drive/MyDrive/sro_minitaur/mpop_ic \
        --pairs directed_forgetting>recent_probes,kirby>discount_titrate --max-seq-len 6144
"""
import argparse, json, os, sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.model.incontext import incontext_response_nll
from sro_transfer.model.masking import RESP_SPAN
from sro_transfer.runtime import get_model
from sro_transfer.stats import paired_report
from sro_transfer.utils import load_config, load_tasks


def marginal(text):
    """Per-person response distribution from <<...>> content (no GPU)."""
    resp = [m.group(1).strip() for m in RESP_SPAN.finditer(text)]
    c = Counter(resp); tot = sum(c.values())
    return {k: v / tot for k, v in c.items()} if tot else {}


def l1(p, q):
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop_ic")
    ap.add_argument("--pairs", default="directed_forgetting>recent_probes,kirby>discount_titrate")
    ap.add_argument("--max-seq-len", type=int, default=6144)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed, rdir = cfg["split"]["seed"], cfg["paths"]["results"]
    tax = load_tasks()
    universe = sorted(set().union(*[set(load_sessions(cfg["paths"]["nl_dir"], t, "complete"))
                                    for t in tax["subsets"]["starting_subset"]]))
    split = make_splits(universe, [], cfg["split"]["heldout_frac"], seed)
    heldout = set(split.heldout)
    model, tok = get_model(cfg, args.mpop)
    rng = np.random.RandomState(seed)

    out = Path(rdir) / "baserate_control"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for pair in args.pairs.split(","):
        a, b = pair.split(">")
        A = load_sessions(cfg["paths"]["nl_dir"], a, "complete")
        B = load_sessions(cfg["paths"]["nl_dir"], b, "complete")
        held = [w for w in heldout if w in A and w in B]
        if args.limit:
            held = held[: args.limit]
        margA = {w: marginal(A[w]) for w in held}
        # near-degeneracy of the TARGET (how base-rate-exposed this pair is)
        modal = []
        for w in held:
            mb = marginal(B[w]); modal.append(max(mb.values()) if mb else 1.0)
        rows = []
        for i, p in enumerate(held, 1):
            others = [q for q in held if q != p]
            qstar = min(others, key=lambda q: l1(margA[p], margA[q]))   # marginal-matched
            qrand = others[rng.randint(len(others))]                    # random
            rows.append({
                "wid": p,
                "floor": incontext_response_nll(model, tok, B[p], None, args.max_seq_len),
                "real": incontext_response_nll(model, tok, B[p], A[p], args.max_seq_len),
                "matched": incontext_response_nll(model, tok, B[p], A[qstar], args.max_seq_len),
                "rand": incontext_response_nll(model, tok, B[p], A[qrand], args.max_seq_len),
                "qstar": qstar, "qrand": qrand,
                "l1_to_matched": float(l1(margA[p], margA[qstar])),
            })
            if i % 20 == 0:
                print(f"  {pair}: {i}/{len(held)}")
        real = [r["real"] for r in rows]
        report[pair] = {
            "n": len(rows),
            "target_median_modal_fraction": float(np.median(modal)),
            "target_frac_near_degenerate(>0.9)": float(np.mean(np.array(modal) > 0.9)),
            "real_vs_floor": paired_report(real, [r["floor"] for r in rows], seed=seed),
            "real_vs_random_shuffled": paired_report(real, [r["rand"] for r in rows], seed=seed),
            "real_vs_MARGINAL_MATCHED": paired_report(real, [r["matched"] for r in rows], seed=seed),
        }
        (out / f"{a}__{b}.json").write_text(json.dumps({"rows": rows, "report": report[pair]}, indent=2))
        m = report[pair]["real_vs_MARGINAL_MATCHED"]
        print(f"\n{pair}: real vs MARGINAL-MATCHED  mean_diff={m['mean_diff']:+.4f} "
              f"perm_p={m['perm_p']:.1e}  frac_below={m['frac_real_below']:.2f}  "
              f"(target modal {report[pair]['target_median_modal_fraction']:.2f})")
        print(f"   => real<matched means person-specific BEYOND base rate\n")

    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
