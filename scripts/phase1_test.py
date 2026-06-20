#!/usr/bin/env python
"""PHASE 1 TEST -- how well does M_pop predict held-out behavior, PER TASK?

On the held-out test subjects, for every task, report M_pop's:
  - response-token accuracy  (model argmax == the human's actual response)
  - mean response NLL
  - a majority-response baseline (most frequent response), so accuracy is readable

This is the real Phase-1 evaluation: is the population floor a good model of SRO
behavior on each task? (No individual info yet -- that's Phase 2+.)

    python scripts/phase1_test.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --subset starting_subset
Smoke: add --limit 15
"""
import argparse, json, os, sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.model.masking import build_labels
from sro_transfer.runtime import get_model, get_splits
from sro_transfer.utils import load_config, load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--subset", default="starting_subset")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #subjects/task (smoke)")
    ap.add_argument("--out", default="/content/drive/MyDrive/sro_minitaur/phase1_test.json")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    cfg = load_config(args.config)
    max_len = cfg["model"]["max_seq_len"]
    tasks = args.tasks or load_tasks()["subsets"][args.subset]
    model, tok = get_model(cfg, args.mpop)

    @torch.no_grad()
    def session_stats(text):
        """Return (sum_nll, n_correct, n_tok, [actual response token ids])."""
        e = build_labels(text, tok, max_len)
        ids = torch.tensor([e["input_ids"]], device=model.device)
        att = torch.tensor([e["attention_mask"]], device=model.device)
        lab = torch.tensor(e["labels"], device=model.device)            # [L]
        logits = model(input_ids=ids, attention_mask=att).logits[0]     # [L, V]
        # causal shift: token at pos i is predicted from logits[i-1]
        sl, tl = logits[:-1], lab[1:]
        m = tl != -100
        if m.sum() == 0:
            return 0.0, 0, 0, []
        sl, tl = sl[m], tl[m]
        nll = F.cross_entropy(sl.float(), tl, reduction="sum")
        correct = (sl.argmax(-1) == tl).sum()
        return float(nll), int(correct), int(m.sum()), tl.tolist()

    rows = []
    for task in tasks:
        split, tgt = get_splits(cfg, task)
        held = [w for w in split.heldout if w in tgt]
        if args.limit:
            held = held[: args.limit]
        s_nll = n_cor = n_tok = 0
        all_tokens = Counter()
        for w in held:
            nll, cor, ntok, toks = session_stats(tgt[w])
            s_nll += nll; n_cor += cor; n_tok += ntok
            all_tokens.update(toks)
        if n_tok == 0:
            continue
        majority = max(all_tokens.values()) / sum(all_tokens.values())
        rows.append({
            "task": task, "n_subj": len(held), "n_resp_tokens": n_tok,
            "accuracy": round(n_cor / n_tok, 4),
            "majority_baseline": round(majority, 4),
            "mean_nll": round(s_nll / n_tok, 4),
        })
        print(f"  {task:<28} acc={n_cor/n_tok:.3f} (base {majority:.3f})  nll={s_nll/n_tok:.3f}  n={len(held)}")

    overall = {
        "phase": "1-test",
        "macro_accuracy": round(sum(r["accuracy"] for r in rows) / max(len(rows), 1), 4),
        "per_task": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(overall, indent=2))
    print(f"\nmacro accuracy over tasks = {overall['macro_accuracy']:.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
