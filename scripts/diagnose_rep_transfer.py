#!/usr/bin/env python
"""DIAGNOSTIC -- is cross-task person signal even IN the LM reps? (no training)

Localizes why Phase 2 is flat:
  - per person, mean-pool their source-task and target-task trial-reps -> 2 vectors
  - fit a linear map source->target on TRAIN people (projects out task content,
    keeps the person-consistent part)
  - on HELD-OUT people, identify: does W*source_rep pick the right person's
    target_rep out of K candidates?

Above chance  => person signal IS in the reps; the soft-prompt injection is the
                 bottleneck -> worth building a stronger mechanism (FiLM).
~chance       => these choice-level reps don't carry transferable person info
                 -> not a mechanism bug; rethink reps or accept the null.

    python scripts/diagnose_rep_transfer.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --source directed_forgetting --target recent_probes
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sro_transfer.runtime import get_model, get_reps, get_splits
from sro_transfer.utils import load_config


def _mean_vecs(reps):
    return {w: r.float().mean(0).cpu().numpy() for w, r in reps.items()}


def _identify(pred, true_vecs, ids, K, seed):
    rng = np.random.RandomState(seed)
    T = np.stack([true_vecs[w] for w in ids])
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    ranks = []
    for i, w in enumerate(ids):
        p = pred[i] / (np.linalg.norm(pred[i]) + 1e-8)
        others = [j for j in range(len(ids)) if j != i]
        cand = [i] + list(rng.choice(others, size=min(K - 1, len(others)), replace=False))
        sims = T[cand] @ p
        order = np.argsort(-sims)
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
    seed = cfg["split"]["seed"]
    model, tok = get_model(cfg, args.mpop)

    split, tgt = get_splits(cfg, args.target)
    src_reps = get_reps(model, tok, cfg, args.source)
    tgt_reps = get_reps(model, tok, cfg, args.target)
    S, Tv = _mean_vecs(src_reps), _mean_vecs(tgt_reps)

    common = [w for w in (set(S) & set(Tv))]
    train = [w for w in split.train if w in common]
    held = [w for w in split.heldout if w in common]
    print(f"train={len(train)} heldout={len(held)}  {args.source}->{args.target}")

    Xtr = np.stack([S[w] for w in train]); Ytr = np.stack([Tv[w] for w in train])
    Xte = np.stack([S[w] for w in held])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    ridge = Ridge(alpha=100.0).fit(Xtr, Ytr)
    pred = ridge.predict(Xte)

    learned = _identify(pred, Tv, held, args.K, seed)
    raw = _identify(Xte, Tv, held, args.K, seed)   # raw source-rep, no mapping
    out = {"phase": "diagnose-rep-transfer", "source": args.source, "target": args.target,
           "linear_map_identification": learned, "raw_rep_identification": raw}
    print(json.dumps(out, indent=2))
    se = (learned["chance_top1"] * (1 - learned["chance_top1"]) / learned["n"]) ** 0.5
    verdict = ("signal IS in reps -> build FiLM" if learned["top1"] - learned["chance_top1"] > 2 * se
               else "~chance -> reps lack transferable person info")
    print(f"\nlinear-map top1={learned['top1']:.3f} vs chance {learned['chance_top1']:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
