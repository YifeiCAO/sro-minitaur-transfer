#!/usr/bin/env python
"""Phase 2 + 3/4 for ONE source->target pair.

Trains the person-encoder E (source-task trials -> z) + soft-prompt injection
into the FROZEN M_pop to predict the target task, then evaluates:
  - NLL contrast: floor vs real-z vs shuffled-z   (Phase 4)
  - cross-task identification                       (Phase 3)

Only E + the injector are trained; M_pop stays frozen (gradient flows through it
to the soft prefix but its weights never update). Start on the strongest 0c cell:

    python scripts/train_person_transfer.py \
        --mpop /content/drive/MyDrive/sro_minitaur/mpop \
        --source kirby --target discount_titrate \
        --out results/transfer_kirby_discount

Smoke-test first with --limit 20 (a handful of subjects, ~10 min) to shake out
device/dtype/memory issues before the full ~1-2h run.

NOTE: first GPU version of the novel parts (per-trial reps + soft-prompt
backprop through a 4-bit frozen base). Expect to iterate on a real GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.data import load_sessions, make_splits
from sro_transfer.eval.identification import identification_report, identify
from sro_transfer.eval.nll import nll_floor_real_shuffled, summarize
from sro_transfer.model.masking import build_labels
from sro_transfer.model.mpop import load_mpop
from sro_transfer.model.transfer_model import TransferConfig, build_transfer_model
from sro_transfer.model.trial_reps import build_or_load_reps
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
    ap.add_argument("--limit", type=int, default=None, help="cap #subjects (smoke test)")
    ap.add_argument("--out", default="results/transfer")
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config)
    pt = cfg["person_transfer"]
    epochs = args.epochs or pt["epochs"]
    lr = args.lr or pt["lr"]
    max_len = cfg["model"]["max_seq_len"]
    dev = "cuda"

    # ---- frozen M_pop --------------------------------------------------
    model, tok = load_mpop(cfg, args.mpop)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass

    # ---- data + splits -------------------------------------------------
    nl = cfg["paths"]["nl_dir"]
    src_sessions = load_sessions(nl, args.source, "complete")
    tgt_sessions = load_sessions(nl, args.target, "complete")
    retest = list(load_sessions(nl, args.target, "retest"))
    split = make_splits(list(tgt_sessions), retest,
                        cfg["split"]["heldout_frac"], cfg["split"]["seed"])
    print("split:", split.summary())

    # ---- candidate subjects (have a target session), then extract reps -
    cand_train = [w for w in split.train if w in tgt_sessions]
    cand_held = [w for w in split.heldout if w in tgt_sessions]
    if args.limit:
        cand_train, cand_held = cand_train[: args.limit], cand_held[: args.limit]
    needed = set(cand_train) | set(cand_held)
    src_needed = {w: src_sessions[w] for w in needed if w in src_sessions}
    suffix = f"_smoke{args.limit}" if args.limit else ""
    reps_fp = Path(cfg["paths"]["results"]) / "trial_reps" / f"{args.source}{suffix}.pt"
    reps = build_or_load_reps(model, tok, src_needed, reps_fp, max_len)

    # ---- transfer model (E + injector around frozen M_pop) ------------
    hidden = model.config.hidden_size
    tm = build_transfer_model(
        model, tok,
        TransferConfig(z_dim=pt["z_dim"], n_soft_tokens=pt["n_soft_tokens"], d_trial=hidden),
    )
    tm.encoder.to(dev).float()
    tm.injector.to(dev).float()
    opt = torch.optim.AdamW(
        list(tm.encoder.parameters()) + list(tm.injector.parameters()), lr=lr
    )

    def _ids(text):
        e = build_labels(text, tok, max_len)
        return (
            torch.tensor([e["input_ids"]], device=dev),
            torch.tensor([e["attention_mask"]], device=dev),
            torch.tensor([e["labels"]], device=dev),
        )

    def encode(wid):  # grad-on; for training
        r = reps[wid].unsqueeze(0).float().to(dev)        # [1, T, H]
        return tm.encode_person(r)                         # [1, z_dim]

    train_ids = [w for w in cand_train if w in reps]
    held_ids = [w for w in cand_held if w in reps]
    print(f"train={len(train_ids)} heldout={len(held_ids)}  pair={args.source}->{args.target}")

    # ---- train ---------------------------------------------------------
    tm.encoder.train(); tm.injector.train()
    rng = __import__("random").Random(cfg["split"]["seed"])
    for ep in range(epochs):
        rng.shuffle(train_ids)
        opt.zero_grad()
        running = 0.0
        for i, w in enumerate(train_ids, 1):
            z = encode(w)
            ids, att, lab = _ids(tgt_sessions[w])
            loss = tm.target_nll(ids, att, lab, z) / args.grad_accum
            loss.backward()
            running += float(loss) * args.grad_accum
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(tm.encoder.parameters()) + list(tm.injector.parameters()), 1.0
                )
                opt.step(); opt.zero_grad()
            if i % 50 == 0:
                print(f"  epoch {ep} {i}/{len(train_ids)} loss={running/i:.4f}")
        print(f"epoch {ep} mean train loss = {running/max(len(train_ids),1):.4f}")

    # ---- eval (Phase 3 + 4) -------------------------------------------
    tm.encoder.eval(); tm.injector.eval()

    @torch.no_grad()
    def z_of(w):
        return tm.encode_person(reps[w].unsqueeze(0).float().to(dev))

    @torch.no_grad()
    def floor_score(text, z=None):
        ids, att, lab = _ids(text)
        return float(model(input_ids=ids, attention_mask=att, labels=lab).loss)

    @torch.no_grad()
    def transfer_score(text, z):
        ids, att, lab = _ids(text)
        return float(tm.target_nll(ids, att, lab, z))

    held_sessions = {w: tgt_sessions[w] for w in held_ids}

    nll_df = nll_floor_real_shuffled(
        held_sessions, z_of, floor_score, transfer_score,
        n_shuffle=cfg["eval"]["n_shuffle_controls"], seed=cfg["split"]["seed"],
    )
    nll_sum = summarize(nll_df)
    id_res = identify(held_sessions, z_of, transfer_score,
                      K=cfg["eval"]["identification_K"], seed=cfg["split"]["seed"])
    id_rep = identification_report(id_res)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder": tm.encoder.state_dict(), "injector": tm.injector.state_dict()},
        out / "transfer_heads.pt",
    )
    report = {
        "source": args.source, "target": args.target,
        "n_train": len(train_ids), "n_heldout": len(held_ids),
        "nll": nll_sum, "identification": id_rep,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    nll_df.to_csv(out / "nll_per_subject.csv", index=False)

    print("\n=== RESULT ===")
    print(json.dumps(report, indent=2))
    print(
        "\nKEY: identification top1 >> "
        f"{id_rep.get('chance_top1', float('nan')):.3f} (chance) and "
        "nll.mean_real_minus_shuffled < 0 => person-specific transfer."
    )


if __name__ == "__main__":
    main()
