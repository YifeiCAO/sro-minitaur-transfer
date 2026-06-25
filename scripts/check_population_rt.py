#!/usr/bin/env python
"""SANITY CHECK: does M_pop_RT actually predict reaction time at the POPULATION
level? Before any individual-residual analysis, the population RT model must beat
base rate -- otherwise the "RT residual" is just noise.

On held-out subjects, at each response token, compare the model to a base-rate
(unigram) baseline, split into CHOICE tokens vs RT-bin tokens:
  - mean NLL (model vs base-rate; model lower = predicts beyond base rate)
  - top-1 accuracy
RT bins are deciles -> base-rate accuracy on the bin digit is ~0.10.

    python scripts/check_population_rt.py --mpop /content/drive/MyDrive/sro_minitaur/mpop_rt \
        --nl-dir /content/drive/MyDrive/sro_minitaur/output_nl_rt --subset all --limit 40
"""
import argparse, json, math, os, sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.model.masking import build_labels
from sro_transfer.model.surprise import _RT_TOK
from sro_transfer.runtime import get_model
from sro_transfer.utils import load_config, load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop_rt")
    ap.add_argument("--nl-dir", default="/content/drive/MyDrive/sro_minitaur/output_nl_rt")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--limit", type=int, default=40, help="held-out subjects per task")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    cfg = load_config(args.config)
    cfg["paths"]["nl_dir"] = args.nl_dir
    seed = cfg["split"]["seed"]
    tax = load_tasks()
    tasks = sorted(tax["tasks"]) if args.subset == "all" else tax["subsets"][args.subset]
    model, tok = get_model(cfg, args.mpop)

    # collect per-token records: (task, is_rt, nll, correct, token_id)
    rows = []
    for t in tasks:
        sess = load_sessions(args.nl_dir, t, "complete")
        if not sess:
            continue
        split = make_splits(list(sess), [], cfg["split"]["heldout_frac"], seed)
        held = [w for w in split.heldout if w in sess][: args.limit]
        for w in held:
            e = build_labels(sess[w], tok, args.max_seq_len)
            ids = torch.tensor([e["input_ids"]], device=model.device)
            with torch.no_grad():
                logits = model(input_ids=ids).logits[0]
            sl = logits[:-1].float()
            tl = torch.tensor(e["labels"], device=model.device)[1:]
            mask = tl != -100
            if int(mask.sum()) == 0:
                continue
            pos = mask.nonzero().squeeze(-1)
            logp = F.log_softmax(sl[pos], dim=-1)
            gold = tl[pos]
            nll = (-logp[torch.arange(len(pos), device=logp.device), gold]).tolist()
            correct = (logp.argmax(-1) == gold).tolist()
            pos_l = pos.tolist(); gold_l = gold.tolist()
            i = 0
            while i < len(pos_l):                       # group contiguous -> one response
                j = i
                while j + 1 < len(pos_l) and pos_l[j + 1] == pos_l[j] + 1:
                    j += 1
                is_rt = bool(_RT_TOK.match(tok.decode(gold_l[i:j + 1]).strip().replace(" ", "")))
                for k in range(i, j + 1):
                    rows.append((t, is_rt, nll[k], correct[k], gold_l[k]))
                i = j + 1
        print(f"  {t}: {len([r for r in rows if r[0]==t])} tokens")

    def summarize(rs, label):
        if not rs:
            print(f"  {label}: no tokens"); return None
        nll = np.array([r[2] for r in rs]); acc = np.mean([r[3] for r in rs])
        freq = Counter(r[4] for r in rs); n = len(rs)
        base_nll = np.mean([-math.log(freq[r[4]] / n) for r in rs])
        base_acc = max(freq.values()) / n
        print(f"  {label:8s} n={n:6d}  model NLL={nll.mean():.3f}  base-rate NLL={base_nll:.3f}  "
              f"| acc={acc:.3f}  base-rate acc={base_acc:.3f}")
        return {"n": n, "model_nll": float(nll.mean()), "base_nll": float(base_nll),
                "acc": float(acc), "base_acc": float(base_acc)}

    print("\n=== population prediction on held-out subjects ===")
    rt_rows = [r for r in rows if r[1]]
    ch_rows = [r for r in rows if not r[1]]
    res = {"choice": summarize(ch_rows, "CHOICE"), "rt": summarize(rt_rows, "RT")}
    if res["rt"]:
        good = res["rt"]["model_nll"] < res["rt"]["base_nll"] - 0.02
        print(f"\nVERDICT: M_pop_RT predicts RT beyond base rate? "
              f"{'YES' if good else 'NO / marginal'} "
              f"(model {res['rt']['model_nll']:.3f} vs base {res['rt']['base_nll']:.3f})")
        print("If NO -> the RT residual is mostly noise; the individual-RT story is shaky.")
    Path(cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["results"]) / "population_rt_check.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
