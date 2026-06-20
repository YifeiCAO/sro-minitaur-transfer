#!/usr/bin/env python
"""DIAGNOSTIC -- does the SURPRISE profile transfer across tasks? (training-free)

The hidden-state rep was person-invariant (encodes population expectation). The
individual signal in a choice-only FM is the residual = how surprised M_pop is by
each person's choices. This tests whether THAT transfers:

  per person -> (surprise, entropy) profile on source & target -> 17-d summary
  -> linear-map source->target on TRAIN -> held-out identification.

Above chance => the surprise residual carries transferable person signal
                -> rebuild Phase 2 with surprise reps (worth it).
~chance       => even the residual doesn't transfer at the choice level
                -> the choice-only null is real (the RT-limited story).

    python scripts/diagnose_surprise_transfer.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --source directed_forgetting --target recent_probes
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sro_transfer.data import load_sessions
from sro_transfer.model.surprise import build_or_load_profiles, summarize_profile
from sro_transfer.runtime import get_model, get_splits
from sro_transfer.utils import load_config


def _identify(pred, true_vecs, ids, K, seed):
    rng = np.random.RandomState(seed)
    T = np.stack([true_vecs[w] for w in ids])
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    ranks = []
    for i in range(len(ids)):
        p = pred[i] / (np.linalg.norm(pred[i]) + 1e-8)
        others = [j for j in range(len(ids)) if j != i]
        cand = [i] + list(rng.choice(others, size=min(K - 1, len(others)), replace=False))
        order = np.argsort(-(T[cand] @ p))
        ranks.append(int(np.where(np.array(cand)[order] == i)[0][0]) + 1)
    ranks = np.array(ranks)
    return {"top1": float((ranks == 1).mean()), "mean_rank": float(ranks.mean()),
            "chance_top1": 1.0 / K, "n": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()
    from sklearn.linear_model import Ridge

    cfg = load_config(args.config)
    seed, max_len = cfg["split"]["seed"], cfg["model"]["max_seq_len"]
    rdir = cfg["paths"]["results"]
    model, tok = get_model(cfg, args.mpop)
    split, _ = get_splits(cfg, args.target)

    src = load_sessions(cfg["paths"]["nl_dir"], args.source, "complete")
    tgt = load_sessions(cfg["paths"]["nl_dir"], args.target, "complete")
    Ps = build_or_load_profiles(model, tok, src, Path(rdir) / "surprise" / f"{args.source}.pt", max_len)
    Pt = build_or_load_profiles(model, tok, tgt, Path(rdir) / "surprise" / f"{args.target}.pt", max_len)
    S = {w: summarize_profile(p) for w, p in Ps.items()}
    T = {w: summarize_profile(p) for w, p in Pt.items()}

    common = set(S) & set(T)
    train = [w for w in split.train if w in common]
    held = [w for w in split.heldout if w in common]
    print(f"train={len(train)} heldout={len(held)}  {args.source}->{args.target}")

    Xtr = np.stack([S[w] for w in train]); Ytr = np.stack([T[w] for w in train])
    Xte = np.stack([S[w] for w in held])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6
    Ytr_n = (Ytr - ymu) / ysd
    Tn = {w: (T[w] - ymu) / ysd for w in held}

    pred = Ridge(alpha=10.0).fit(Xtr, Ytr_n).predict(Xte)
    learned = _identify(pred, Tn, held, args.K, seed)
    raw = _identify(Xte, Tn, held, args.K, seed)
    out = {"phase": "diagnose-surprise-transfer", "source": args.source, "target": args.target,
           "linear_map_identification": learned, "raw_summary_identification": raw}
    print(json.dumps(out, indent=2))
    se = (learned["chance_top1"] * (1 - learned["chance_top1"]) / learned["n"]) ** 0.5
    verdict = ("surprise transfers -> rebuild Phase 2 with surprise reps"
               if learned["top1"] - learned["chance_top1"] > 2 * se
               else "~chance -> choice-only residual does not transfer (real null direction)")
    print(f"\nlinear-map top1={learned['top1']:.3f} vs chance {learned['chance_top1']:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
