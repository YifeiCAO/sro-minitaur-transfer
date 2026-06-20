"""Shared runtime helpers so each phase script (and notebook) stays thin.

Every phase: load what it needs from Drive -> do its work -> save to Drive.
This module centralizes model/reps/heads loading and the scoring closures.
"""
from __future__ import annotations

from pathlib import Path

from .data import load_sessions, make_splits
from .model.masking import build_labels
from .model.mpop import load_mpop
from .model.transfer_model import TransferConfig, build_transfer_model
from .model.trial_reps import build_or_load_reps


def get_model(cfg, mpop_dir):
    """Load frozen M_pop (base + trained LoRA adapter) + tokenizer."""
    return load_mpop(cfg, mpop_dir)


def get_splits(cfg, target):
    """Return (SubjectSplit, target_sessions_dict) for a target task."""
    nl = cfg["paths"]["nl_dir"]
    tgt = load_sessions(nl, target, "complete")
    retest = list(load_sessions(nl, target, "retest"))
    split = make_splits(list(tgt), retest, cfg["split"]["heldout_frac"], cfg["split"]["seed"])
    return split, tgt


def get_reps(model, tok, cfg, source, subjects=None, smoke=None):
    """Per-trial reps for a source task (cached to Drive results dir)."""
    src = load_sessions(cfg["paths"]["nl_dir"], source, "complete")
    if subjects is not None:
        src = {w: src[w] for w in subjects if w in src}
    suffix = f"_smoke{smoke}" if smoke else ""
    fp = Path(cfg["paths"]["results"]) / "trial_reps" / f"{source}{suffix}.pt"
    return build_or_load_reps(model, tok, src, fp, cfg["model"]["max_seq_len"])


def build_heads(model, tok, cfg):
    """Build the (untrained) person-encoder + injector around frozen M_pop."""
    pt = cfg["person_transfer"]
    tm = build_transfer_model(
        model, tok,
        TransferConfig(z_dim=pt["z_dim"], n_soft_tokens=pt["n_soft_tokens"],
                       d_trial=model.config.hidden_size),
    )
    tm.encoder.to(model.device).float()
    tm.injector.to(model.device).float()
    return tm


def save_heads(tm, path):
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": tm.encoder.state_dict(),
                "injector": tm.injector.state_dict()}, path)


def load_heads(tm, path):
    import torch
    sd = torch.load(path, map_location=tm.mpop.device)
    tm.encoder.load_state_dict(sd["encoder"])
    tm.injector.load_state_dict(sd["injector"])
    tm.encoder.eval(); tm.injector.eval()
    return tm


def make_scorers(model, tm, tok, reps, max_len, dev=None):
    """Return (z_of, floor_score, transfer_score) closures for eval."""
    import torch
    dev = dev or model.device

    def _ids(text):
        e = build_labels(text, tok, max_len)
        return (torch.tensor([e["input_ids"]], device=dev),
                torch.tensor([e["attention_mask"]], device=dev),
                torch.tensor([e["labels"]], device=dev))

    @torch.no_grad()
    def z_of(wid):
        return tm.encode_person(reps[wid].unsqueeze(0).float().to(dev))

    @torch.no_grad()
    def floor_score(text, z=None):
        ids, att, lab = _ids(text)
        return float(model(input_ids=ids, attention_mask=att, labels=lab).loss)

    @torch.no_grad()
    def transfer_score(text, z):
        ids, att, lab = _ids(text)
        return float(tm.target_nll(ids, att, lab, z))

    return z_of, floor_score, transfer_score
