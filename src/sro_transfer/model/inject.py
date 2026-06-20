"""Phase 2 (part 2) -- inject z into the frozen M_pop.

Default pathway is a *soft prompt*: z is mapped to ``n_soft_tokens`` virtual
embedding vectors that are prepended to the target session's input embeddings.
The frozen LM then conditions its response-token predictions on the person
embedding. FiLM and an extra LoRA are alternative pathways (same contract).

The shuffled-z control swaps z for another person's embedding of identical norm
and dimensionality, so any benefit must be person-specific, not extra capacity.
"""
from __future__ import annotations


def build_soft_prompt_injector(z_dim: int, hidden_size: int, n_soft_tokens: int = 8):
    import torch.nn as nn

    class SoftPromptInjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.n = n_soft_tokens
            self.h = hidden_size
            self.proj = nn.Sequential(
                nn.Linear(z_dim, hidden_size), nn.GELU(),
                nn.Linear(hidden_size, n_soft_tokens * hidden_size),
            )

        def prefix_embeds(self, z):
            # z: [B, z_dim] -> [B, n_soft_tokens, hidden_size]
            return self.proj(z).view(z.shape[0], self.n, self.h)

        def forward(self, z, token_embeds, attention_mask=None):
            """Prepend z-derived virtual tokens to the LM's input embeddings.

            token_embeds: [B, L, H] from model.get_input_embeddings()(input_ids)
            Returns (inputs_embeds, attention_mask, n_prefix) so the caller can
            shift the label mask by n_prefix (prefix tokens are never targets).
            """
            import torch

            # cast the (fp32) prefix to the LM's embedding dtype (bf16/fp16)
            pre = self.prefix_embeds(z).to(token_embeds.dtype)   # [B, n, H]
            inputs_embeds = torch.cat([pre, token_embeds], dim=1)
            if attention_mask is not None:
                pad = torch.ones(
                    attention_mask.shape[0], self.n,
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([pad, attention_mask], dim=1)
            return inputs_embeds, attention_mask, self.n

    return SoftPromptInjector()
