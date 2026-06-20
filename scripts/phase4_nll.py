#!/usr/bin/env python
"""PHASE 4 -- NLL contrast: real-z vs floor vs shuffled-z (loads from Drive).

    python scripts/phase4_nll.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --source kirby --target discount_titrate
Use the SAME --source/--limit as phase2 so the trial-rep cache matches.
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.eval.nll import nll_floor_real_shuffled, summarize
from sro_transfer.runtime import (build_heads, get_model, get_reps, get_splits,
                                   load_heads, make_scorers)
from sro_transfer.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base", default="/content/drive/MyDrive/sro_minitaur")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tok = get_model(cfg, args.mpop)
    split, tgt = get_splits(cfg, args.target)
    cand_held = [w for w in split.heldout if w in tgt]
    if args.limit:
        cand_held = cand_held[: args.limit]
    reps = get_reps(model, tok, cfg, args.source, subjects=cand_held, smoke=args.limit)

    tm = build_heads(model, tok, cfg)
    heads_fp = Path(args.base) / "transfer" / f"{args.source}_{args.target}" / "heads.pt"
    load_heads(tm, heads_fp)

    z_of, floor, transfer = make_scorers(model, tm, tok, reps, cfg["model"]["max_seq_len"])
    held_sessions = {w: tgt[w] for w in cand_held if w in reps}
    df = nll_floor_real_shuffled(held_sessions, z_of, floor, transfer,
                                 n_shuffle=cfg["eval"]["n_shuffle_controls"],
                                 seed=cfg["split"]["seed"])
    summary = summarize(df)
    summary.update({"phase": "4", "source": args.source, "target": args.target})

    out = heads_fp.parent / "nll.json"
    out.write_text(json.dumps(summary, indent=2))
    df.to_csv(heads_fp.parent / "nll_per_subject.csv", index=False)
    print(json.dumps(summary, indent=2))
    print("\nreal < shuffled (mean_real_minus_shuffled < 0) => person-specific transfer.")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
