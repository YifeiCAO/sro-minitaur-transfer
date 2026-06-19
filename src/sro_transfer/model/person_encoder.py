"""Phase 2 (part 1) -- per-source-task person-encoder E_A: trials -> z_p^A.

Permutation-invariant DeepSets pooling over a person's trials on source task A.
Each trial is represented by a fixed-width vector ``d_in`` (e.g. mean-pooled
frozen-M_pop hidden states over that trial's tokens, or handcrafted per-trial
features). The encoder maps the *set* of a person's trials to a single
embedding z that is later injected into the frozen M_pop.

Only E_A and the injection pathway are trained; M_pop stays frozen. By
construction z can only carry information about how this person *deviates* from
the population, which is what makes the shuffled-z control meaningful.
"""
from __future__ import annotations


def build_person_encoder(d_in: int, z_dim: int, hidden: int = 256):
    import torch.nn as nn

    class DeepSetsEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
            )
            self.rho = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, z_dim),
            )

        def forward(self, trials, mask=None):
            # trials: [B, T, d_in]; mask: [B, T] (1 = real trial)
            h = self.phi(trials)                       # [B, T, hidden]
            if mask is not None:
                h = h * mask.unsqueeze(-1)
                pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                pooled = h.mean(1)
            return self.rho(pooled)                     # [B, z_dim]

    return DeepSetsEncoder()
