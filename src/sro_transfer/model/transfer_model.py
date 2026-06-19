"""Phase 2 (part 3) -- the transfer model: frozen M_pop + E_A + injection.

Ties together:
  * frozen M_pop (from Phase 1)               -- never updated
  * person-encoder E_A (person_encoder.py)    -- trained
  * soft-prompt injector (inject.py)          -- trained

Training objective: minimize response-token NLL on a *target* task B session,
conditioned on z computed from the same person's *source* task A trials. The
improvement over the floor (M_pop with no z) is the A->B individual transfer;
the shuffled-z control bounds how much of it is person-specific.

This is the integration scaffold. The two seams flagged TODO are the only
model-specific wiring: (1) producing per-trial representations for E_A, and
(2) the forward that splices the soft prefix into the frozen LM and reads off
masked response NLL. Both are small once a concrete M_pop checkpoint is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransferConfig:
    z_dim: int = 64
    n_soft_tokens: int = 8
    d_trial: int = 4096          # per-trial representation width (LM hidden size)


def build_transfer_model(mpop_model, tokenizer, cfg: TransferConfig):
    """Assemble E_A + injector around a frozen M_pop.

    Returns an object exposing:
      encode_person(source_trials) -> z
      target_nll(target_text, z)   -> scalar response-NLL
    """
    import torch
    import torch.nn as nn

    from .inject import build_soft_prompt_injector
    from .person_encoder import build_person_encoder

    for p in mpop_model.parameters():       # freeze M_pop
        p.requires_grad_(False)
    hidden = mpop_model.config.hidden_size

    class TransferModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mpop = mpop_model
            self.tok = tokenizer
            self.encoder = build_person_encoder(cfg.d_trial, cfg.z_dim)
            self.injector = build_soft_prompt_injector(cfg.z_dim, hidden, cfg.n_soft_tokens)

        def encode_person(self, trials, mask=None):
            return self.encoder(trials, mask)

        def target_nll(self, input_ids, attention_mask, labels, z):
            # splice z-prefix into the embeddings of the target session
            embed = self.mpop.get_input_embeddings()(input_ids)
            inputs_embeds, attention_mask, n_prefix = self.injector(
                z, embed, attention_mask
            )
            # prefix tokens are never prediction targets
            pad = torch.full(
                (labels.shape[0], n_prefix), -100, dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([pad, labels], dim=1)
            out = self.mpop(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
            )
            return out.loss

    return TransferModel()
