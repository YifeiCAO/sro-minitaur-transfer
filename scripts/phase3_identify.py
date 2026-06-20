#!/usr/bin/env python
"""PHASE 3 -- cross-task identification (loads M_pop + trained heads from Drive).

    python scripts/phase3_identify.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --source kirby --target discount_titrate
Use the SAME --source/--limit as phase2 so the trial-rep cache matches.
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.eval.identification import identification_report, identify
from sro_transfer.runtime import (build_heads, get_model, get_reps, get_splits,
                                   load_heads, make_scorers)
from sro_transfer.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base", default="/content/drive/MyDrive/sro_minitaur")
    args = ap.parse_args()

    cfg = load_config(args.config)
    K = args.K or cfg["eval"]["identification_K"]
    model, tok = get_model(cfg, args.mpop)
    split, tgt = get_splits(cfg, args.target)
    cand_held = [w for w in split.heldout if w in tgt]
    if args.limit:
        cand_held = cand_held[: args.limit]
    reps = get_reps(model, tok, cfg, args.source, subjects=cand_held, smoke=args.limit)

    tm = build_heads(model, tok, cfg)
    heads_fp = Path(args.base) / "transfer" / f"{args.source}_{args.target}" / "heads.pt"
    load_heads(tm, heads_fp)

    z_of, _floor, transfer = make_scorers(model, tm, tok, reps, cfg["model"]["max_seq_len"])
    held_sessions = {w: tgt[w] for w in cand_held if w in reps}
    res = identify(held_sessions, z_of, transfer, K=K, seed=cfg["split"]["seed"])
    rep = identification_report(res)
    rep.update({"phase": "3", "source": args.source, "target": args.target})

    out = heads_fp.parent / "identification.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\ntop1 >> {1/K:.2f} (chance) => cross-task transfer.\nsaved -> {out}")


if __name__ == "__main__":
    main()
