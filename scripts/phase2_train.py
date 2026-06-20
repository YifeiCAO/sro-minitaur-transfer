#!/usr/bin/env python
"""PHASE 2 TRAIN -- train the person-encoder + soft-prompt injection.

Loads frozen M_pop, builds/loads source-task trial-reps, trains E + injector to
make a person's source-task fingerprint predict their target-task behavior, then
SAVES the trained heads to Drive (and loads them back to verify).

    python scripts/phase2_train.py --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --source kirby --target discount_titrate --epochs 3
Smoke test: add --limit 20 --epochs 1
"""
import argparse, os, random, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.model.masking import build_labels
from sro_transfer.runtime import (build_heads, get_model, get_reps, get_splits,
                                   load_heads, save_heads)
from sro_transfer.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base", default="/content/drive/MyDrive/sro_minitaur")
    args = ap.parse_args()
    import torch

    cfg = load_config(args.config)
    pt = cfg["person_transfer"]
    epochs = args.epochs or pt["epochs"]
    lr = args.lr or pt["lr"]
    max_len = cfg["model"]["max_seq_len"]

    model, tok = get_model(cfg, args.mpop)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass

    split, tgt = get_splits(cfg, args.target)
    cand_train = [w for w in split.train if w in tgt]
    cand_held = [w for w in split.heldout if w in tgt]
    if args.limit:
        cand_train, cand_held = cand_train[: args.limit], cand_held[: args.limit]
    needed = set(cand_train) | set(cand_held)
    reps = get_reps(model, tok, cfg, args.source, subjects=needed, smoke=args.limit)
    train_ids = [w for w in cand_train if w in reps]
    print(f"train={len(train_ids)}  pair={args.source}->{args.target}  epochs={epochs}")

    tm = build_heads(model, tok, cfg)
    params = list(tm.encoder.parameters()) + list(tm.injector.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    def _ids(text):
        e = build_labels(text, tok, max_len)
        return (torch.tensor([e["input_ids"]], device=model.device),
                torch.tensor([e["attention_mask"]], device=model.device),
                torch.tensor([e["labels"]], device=model.device))

    tm.encoder.train(); tm.injector.train()
    rng = random.Random(cfg["split"]["seed"])
    for ep in range(epochs):
        rng.shuffle(train_ids)
        opt.zero_grad(); running = 0.0
        for i, w in enumerate(train_ids, 1):
            z = tm.encode_person(reps[w].unsqueeze(0).float().to(model.device))
            ids, att, lab = _ids(tgt[w])
            loss = tm.target_nll(ids, att, lab, z) / args.grad_accum
            loss.backward()
            running += float(loss) * args.grad_accum
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
            if i % 50 == 0:
                print(f"  epoch {ep} {i}/{len(train_ids)} loss={running/i:.4f}")
        print(f"epoch {ep} mean loss = {running/max(len(train_ids),1):.4f}")

    heads_fp = Path(args.base) / "transfer" / f"{args.source}_{args.target}" / "heads.pt"
    save_heads(tm, heads_fp)
    load_heads(tm, heads_fp)  # verify it reloads
    print(f"\nPhase 2 done. heads saved + reloaded OK -> {heads_fp}")


if __name__ == "__main__":
    main()
