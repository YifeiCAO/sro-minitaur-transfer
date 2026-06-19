"""Response-token NLL scoring -- the common currency of Phases 3-4.

Everything downstream (identification, NLL deltas) reduces to: given a target
session and (optionally) a person embedding z, how surprised is the model by the
human's ``<<...>>`` responses? We always reduce to MEAN per-response-token NLL so
sessions of different lengths are comparable.
"""
from __future__ import annotations

from ..model.masking import build_labels


def _mean_response_nll_from_loss(loss) -> float:
    # HF CausalLM with labels already returns mean CE over non-masked tokens.
    return float(loss.detach().cpu())


def make_floor_scorer(model, tokenizer, max_len: int = 4096):
    """score(text, z=None) -> mean response-token NLL under M_pop (z ignored)."""
    import torch

    model.eval()

    @torch.no_grad()
    def score(text: str, z=None) -> float:
        enc = build_labels(text, tokenizer, max_len)
        ids = torch.tensor([enc["input_ids"]], device=model.device)
        att = torch.tensor([enc["attention_mask"]], device=model.device)
        lab = torch.tensor([enc["labels"]], device=model.device)
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        return _mean_response_nll_from_loss(out.loss)

    return score


def make_transfer_scorer(transfer_model, tokenizer, max_len: int = 4096):
    """score(text, z) -> mean response-token NLL under M_pop conditioned on z."""
    import torch

    transfer_model.eval()

    @torch.no_grad()
    def score(text: str, z) -> float:
        enc = build_labels(text, tokenizer, max_len)
        ids = torch.tensor([enc["input_ids"]], device=z.device)
        att = torch.tensor([enc["attention_mask"]], device=z.device)
        lab = torch.tensor([enc["labels"]], device=z.device)
        loss = transfer_model.target_nll(ids, att, lab, z)
        return _mean_response_nll_from_loss(loss)

    return score
