#!/usr/bin/env python
"""In-context cross-task transfer (zero-shot): real-A vs floor vs shuffled-A.

For each held-out person, score their B responses three ways:
  floor     = B alone (no context)
  real-A    = [their own A session] + B
  shuffled-A= [a random other person's A session] + B
real < floor  => A context helps;  real < shuffled => person-specific transfer.
No training. Needs long context (A+B) -> use --max-seq-len 8192.

    python scripts/run_incontext.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --source directed_forgetting --target recent_probes --max-seq-len 8192
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy import stats

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.model.incontext import incontext_response_nll
from sro_transfer.runtime import get_model
from sro_transfer.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nl-dir", default=None, help="override paths.nl_dir (e.g. output_nl_rt)")
    ap.add_argument("--rep", choices=["both", "choice", "rt"], default="both",
                    help="which B-response tokens to score")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.nl_dir:
        cfg["paths"]["nl_dir"] = args.nl_dir
    seed = cfg["split"]["seed"]
    n_shuf = cfg["eval"]["n_shuffle_controls"]
    model, tok = get_model(cfg, args.mpop)

    A = load_sessions(cfg["paths"]["nl_dir"], args.source, "complete")
    B = load_sessions(cfg["paths"]["nl_dir"], args.target, "complete")
    retest = list(load_sessions(cfg["paths"]["nl_dir"], args.target, "retest"))
    split = make_splits(list(B), retest, cfg["split"]["heldout_frac"], seed)
    held = [w for w in split.heldout if w in A and w in B]
    if args.limit:
        held = held[: args.limit]
    rng = np.random.RandomState(seed)
    print(f"heldout={len(held)}  {args.source}->{args.target}  max_len={args.max_seq_len}")

    rows = []
    for i, p in enumerate(held, 1):
        floor = incontext_response_nll(model, tok, B[p], None, args.max_seq_len, rep=args.rep)
        real = incontext_response_nll(model, tok, B[p], A[p], args.max_seq_len, rep=args.rep)
        others = [w for w in held if w != p]
        shuf = np.mean([
            incontext_response_nll(model, tok, B[p], A[q], args.max_seq_len, rep=args.rep)
            for q in rng.choice(others, size=min(n_shuf, len(others)), replace=False)
        ]) if others else np.nan
        rows.append({"wid": p, "floor": floor, "real": real, "shuffled": float(shuf)})
        if i % 20 == 0:
            print(f"  {i}/{len(held)}")

    d = [r for r in rows if all(np.isfinite([r["floor"], r["real"], r["shuffled"]]))]
    real = np.array([r["real"] for r in d]); floor = np.array([r["floor"] for r in d])
    shuf = np.array([r["shuffled"] for r in d])
    out = {
        "phase": "in-context", "source": args.source, "target": args.target, "n": len(d),
        "mean_real_minus_floor": float((real - floor).mean()),
        "p_real_vs_floor": float(stats.ttest_rel(real, floor).pvalue),
        "mean_real_minus_shuffled": float((real - shuf).mean()),
        "p_real_vs_shuffled": float(stats.ttest_rel(real, shuf).pvalue),
        "frac_real_below_shuffled": float((real < shuf).mean()),
    }
    res = Path(cfg["paths"]["results"]) / "incontext"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{args.source}_{args.target}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nreal<floor => A context helps;  real<shuffled (p<.05) => person-specific transfer.")


if __name__ == "__main__":
    main()
