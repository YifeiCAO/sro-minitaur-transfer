#!/usr/bin/env python
"""PHASE 1 TEST -- how well does M_pop predict held-out behavior, PER TASK?

On the held-out test subjects, for every task, report M_pop's:
  - response accuracy: fraction of <<...>> responses the model would reproduce
    EXACTLY (decision-level -- every token of the response right). This is the
    honest number: per-TOKEN accuracy is inflated because multi-token responses
    like <<smaller_sooner>> have trivially predictable continuation subtokens.
  - mean response NLL (per token)
  - a majority-response baseline (most frequent response), so accuracy is readable

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
        """Return (sum_nll, n_tok, [(response_token_tuple, all_tokens_correct)])."""
        e = build_labels(text, tok, max_len)
        ids = torch.tensor([e["input_ids"]], device=model.device)
        att = torch.tensor([e["attention_mask"]], device=model.device)
        lab = torch.tensor(e["labels"], device=model.device)            # [L]
        logits = model(input_ids=ids, attention_mask=att).logits[0]     # [L, V]
        sl, tl = logits[:-1], lab[1:]                                   # causal shift
        preds = sl.argmax(-1)
        m = tl != -100
        sum_nll = float(F.cross_entropy(sl[m].float(), tl[m], reduction="sum")) if m.any() else 0.0
        n_tok = int(m.sum())
        # group consecutive response tokens into one response each
        responses, L, j = [], tl.shape[0], 0
        while j < L:
            if tl[j].item() == -100:
                j += 1; continue
            k = j
            while k < L and tl[k].item() != -100:
                k += 1
            responses.append((tuple(tl[j:k].tolist()), bool((preds[j:k] == tl[j:k]).all())))
            j = k
        return sum_nll, n_tok, responses

    rows = []
    for task in tasks:
        split, tgt = get_splits(cfg, task)
        held = [w for w in split.heldout if w in tgt]
        if args.limit:
            held = held[: args.limit]
        s_nll = n_tok = r_correct = r_total = 0
        r_counter = Counter()
        for w in held:
            nll, ntok, responses = session_stats(tgt[w])
            s_nll += nll; n_tok += ntok
            for lab_ids, correct in responses:
                r_total += 1; r_correct += int(correct); r_counter[lab_ids] += 1
        if r_total == 0:
            continue
        majority = max(r_counter.values()) / r_total
        rows.append({
            "task": task, "n_subj": len(held), "n_responses": r_total,
            "accuracy": round(r_correct / r_total, 4),
            "majority_baseline": round(majority, 4),
            "mean_nll": round(s_nll / max(n_tok, 1), 4),
        })
        print(f"  {task:<28} acc={r_correct/r_total:.3f} (base {majority:.3f})  nll={s_nll/max(n_tok,1):.3f}  n={len(held)}")

    overall = {
        "phase": "1-test", "metric": "per-response (decision-level) accuracy",
        "macro_accuracy": round(sum(r["accuracy"] for r in rows) / max(len(rows), 1), 4),
        "per_task": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(overall, indent=2))
    print(f"\nmacro per-response accuracy = {overall['macro_accuracy']:.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
