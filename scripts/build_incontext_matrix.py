#!/usr/bin/env python
"""In-context transfer MATRIX with the PRIMARY metric (identification) + NLL.

For each (source A, target B) pair, on held-out people:
  - identification: fix person p's B-session, score it under each candidate's
    A-context (p's own + K-1 others'); p's OWN A should make their B least
    surprising -> rank 1. top1 vs chance 1/K. (rank-framing of real<shuffled)
  - NLL contrast: real (own A) vs floor (no A) vs shuffled (others' A).
One set of forwards yields both (the distractor A-contexts ARE the shuffled set).

Uses one consistent union split (matches finetune_incontext). ~50 min/pair at K=10.

    python scripts/build_incontext_matrix.py --mpop /content/drive/MyDrive/sro_minitaur/mpop_ic \
        --pairs within --K 10 --max-seq-len 6144
Split work across GPUs with an explicit list: --pairs kirby>discount_titrate,bickel_titrator>kirby
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.model.incontext import incontext_response_nll
from sro_transfer.runtime import get_model
from sro_transfer.utils import load_config, load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop_ic")
    ap.add_argument("--subset", default="starting_subset")
    ap.add_argument("--pairs", default="within", help="'within' | 'all' | 'A>B,C>D' explicit list")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max-seq-len", type=int, default=6144)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed, rdir = cfg["split"]["seed"], cfg["paths"]["results"]
    tax = load_tasks()
    tasks = sorted(tax["tasks"]) if args.subset == "all" else tax["subsets"][args.subset]
    domain = {t: tax["tasks"][t]["domain"] for t in tasks}
    model, tok = get_model(cfg, args.mpop)

    sess = {t: load_sessions(cfg["paths"]["nl_dir"], t, "complete") for t in tasks}
    universe = sorted(set().union(*[set(s) for s in sess.values()]))
    split = make_splits(universe, [], cfg["split"]["heldout_frac"], seed)
    heldout = set(split.heldout)

    if args.pairs == "within":
        pairs = [(a, b) for a in tasks for b in tasks if a != b and domain[a] == domain[b]]
    elif args.pairs == "all":
        pairs = [(a, b) for a in tasks for b in tasks if a != b]
    else:
        pairs = [tuple(p.replace(":", ">").split(">")) for p in args.pairs.split(",")]  # ':' avoids shell redirect

    out = Path(rdir) / "incontext_matrix"
    out.mkdir(parents=True, exist_ok=True)
    T = pd.DataFrame(index=tasks, columns=tasks, dtype=float)
    # resume: skip pairs already in pair_stats.csv (survives disconnect)
    done = set()
    stats_fp = out / "pair_stats.csv"
    if stats_fp.exists():
        prev = pd.read_csv(stats_fp)
        for _, r in prev.iterrows():
            done.add((r["source"], r["target"])); T.loc[r["source"], r["target"]] = r["id_top1"]
        nll_rows = prev.to_dict("records")
        print(f"resuming: {len(done)} pairs already done")
    else:
        nll_rows = []
    rng = np.random.RandomState(seed)
    for a, b in pairs:
        if (a, b) in done:
            continue
        held = [w for w in heldout if w in sess[a] and w in sess[b]]
        if len(held) < 20:
            continue
        ranks, reals, floors, shufs, raw = [], [], [], [], []
        for p in held:
            floor = incontext_response_nll(model, tok, sess[b][p], None, args.max_seq_len)
            others = [q for q in held if q != p]
            distr = list(rng.choice(others, size=min(args.K - 1, len(others)), replace=False))
            cands = [p] + distr
            nlls = {q: incontext_response_nll(model, tok, sess[b][p], sess[a][q], args.max_seq_len) for q in cands}
            real = nlls[p]
            order = sorted(cands, key=lambda q: nlls[q])
            ranks.append(order.index(p) + 1)
            reals.append(real); floors.append(floor)
            shufs.append(float(np.mean([nlls[q] for q in distr])))
            # raw candidate NLLs -> enables offline permutation / dz / bootstrap / K-id
            raw.append({"wid": p, "floor": floor, "real": real,
                        "distractors": {q: nlls[q] for q in distr}})
        rawdir = out / "raw"; rawdir.mkdir(parents=True, exist_ok=True)
        (rawdir / f"{a}__{b}.json").write_text(json.dumps(raw))
        ranks = np.array(ranks); reals = np.array(reals); floors = np.array(floors); shufs = np.array(shufs)
        top1 = float((ranks == 1).mean())
        T.loc[a, b] = top1
        nll_rows.append({
            "source": a, "target": b, "n": len(held), "id_top1": round(top1, 3),
            "chance": round(1 / args.K, 3), "mean_rank": round(float(ranks.mean()), 2),
            "real_minus_floor": float((reals - floors).mean()),
            "p_real_vs_floor": float(stats.ttest_rel(reals, floors).pvalue),
            "real_minus_shuffled": float((reals - shufs).mean()),
            "p_real_vs_shuffled": float(stats.ttest_rel(reals, shufs).pvalue),
            "frac_real_below_shuffled": float((reals < shufs).mean()),
        })
        r = nll_rows[-1]
        print(f"  {a:>22} -> {b:<22}  id_top1={top1:.3f} (chance {1/args.K:.2f})  "
              f"real-shuf={r['real_minus_shuffled']:+.4f} (p={r['p_real_vs_shuffled']:.1e})")
        # save after EVERY pair (resilient to disconnect)
        T.to_csv(out / "identification_matrix.csv")
        pd.DataFrame(nll_rows).to_csv(out / "pair_stats.csv", index=False)
    within = [r["id_top1"] for r in nll_rows if domain[r["source"]] == domain[r["target"]]]
    across = [r["id_top1"] for r in nll_rows if domain[r["source"]] != domain[r["target"]]]
    summ = {"within_id_top1": float(np.mean(within)) if within else None,
            "across_id_top1": float(np.mean(across)) if across else None,
            "chance": 1 / args.K, "n_pairs": len(nll_rows)}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\nwithin-domain id top1 = {summ['within_id_top1']}  across = {summ['across_id_top1']}  chance = {1/args.K:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
