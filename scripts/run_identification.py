#!/usr/bin/env python
"""Phase 3 -- cross-task identification.

Floor sanity (runnable as soon as M_pop exists): identify with z ignored; should
sit at chance (1/K) and validates the whole scoring/identification plumbing.

    python scripts/run_identification.py --mpop results/mpop --target two_stage_decision

The transfer path (z from a trained person-encoder) is wired below and turns on
once Phase 2 is trained.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.eval import identification_report, identify, make_floor_scorer
from sro_transfer.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="results/mpop")
    ap.add_argument("--target", required=True, help="target task name")
    ap.add_argument("--K", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    K = args.K or cfg["eval"]["identification_K"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.mpop)
    model = AutoModelForCausalLM.from_pretrained(
        args.mpop, torch_dtype=torch.bfloat16, device_map="auto"
    )
    score = make_floor_scorer(model, tok, cfg["model"]["max_seq_len"])

    nl_dir = cfg["paths"]["nl_dir"]
    sessions = load_sessions(nl_dir, args.target, "complete")
    retest_subj = list(load_sessions(nl_dir, args.target, "retest"))
    split = make_splits(list(sessions), retest_subj,
                        cfg["split"]["heldout_frac"], cfg["split"]["seed"])
    heldout = {w: sessions[w] for w in split.heldout if w in sessions}

    res = identify(heldout, z_of=lambda _w: None, score_fn=score, K=K, seed=cfg["split"]["seed"])
    report = identification_report(res)
    report["mode"] = "floor (z ignored, expect ~chance)"
    report["target"] = args.target

    out = Path(cfg["paths"]["results"]) / f"identification_{args.target}_floor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
