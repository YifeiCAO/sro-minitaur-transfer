#!/usr/bin/env python
"""PHASE 1 TEST -- evaluate the frozen M_pop floor (no individual info).

Loads M_pop from Drive, then on held-out subjects reports:
  - mean floor response-NLL on a target task
  - floor identification (z ignored) -> should sit at ~chance = 1/K
This confirms M_pop loads + the scoring/identification plumbing works.

    python scripts/phase1_test.py --mpop /content/drive/MyDrive/sro_minitaur/mpop
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.eval.identification import identification_report, identify
from sro_transfer.model.masking import build_labels
from sro_transfer.runtime import get_model, get_splits
from sro_transfer.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--target", default="two_stage_decision")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="/content/drive/MyDrive/sro_minitaur/phase1_test.json")
    args = ap.parse_args()
    import torch

    cfg = load_config(args.config)
    max_len = cfg["model"]["max_seq_len"]
    model, tok = get_model(cfg, args.mpop)
    split, tgt = get_splits(cfg, args.target)
    held = [w for w in split.heldout if w in tgt]
    if args.limit:
        held = held[: args.limit]
    held_sessions = {w: tgt[w] for w in held}

    @torch.no_grad()
    def floor_score(text, z=None):
        e = build_labels(text, tok, max_len)
        ids = torch.tensor([e["input_ids"]], device=model.device)
        att = torch.tensor([e["attention_mask"]], device=model.device)
        lab = torch.tensor([e["labels"]], device=model.device)
        return float(model(input_ids=ids, attention_mask=att, labels=lab).loss)

    nlls = [floor_score(t) for t in held_sessions.values()]
    mean_nll = sum(nlls) / max(len(nlls), 1)

    res = identify(held_sessions, z_of=lambda _w: None, score_fn=floor_score,
                   K=args.K, seed=cfg["split"]["seed"])
    rep = identification_report(res)

    report = {"phase": "1-test", "target": args.target, "n_heldout": len(held),
              "mean_floor_nll": mean_nll, "floor_identification": rep}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n(floor identification top1 should be ~{1/args.K:.2f} = chance)\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
