"""Phase 2 (input side) -- per-trial person representations from frozen M_pop.

For a session, we mean-pool the last-layer hidden states over each trial's
``<<response>>`` tokens -> one vector per trial. The hidden state at a response
token already encodes the preceding stimulus via attention, so this captures
"how this person responded, in context". These vectors are the input to the
person-encoder E.

The 8B forwards are the expensive part, so results are cached to disk and reused.
"""
from __future__ import annotations

from pathlib import Path

from .masking import response_char_spans


def extract_session_reps(model, tokenizer, text: str, max_len: int):
    """Return a [n_trials, hidden] float16 CPU tensor, or None if no responses fit."""
    import torch

    enc = tokenizer(
        text, truncation=True, max_length=max_len,
        return_offsets_mapping=True, return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states[-1][0]                      # [L, H] last layer

    reps = []
    for s, e in response_char_spans(text):             # inner <<...>> content spans
        idx = [i for i, (a, b) in enumerate(offsets) if a != b and a < e and b > s]
        if idx:
            reps.append(hs[idx].mean(dim=0))
    if not reps:
        return None
    return torch.stack(reps).to(torch.float16).cpu()   # [T, H]


def build_or_load_reps(model, tokenizer, sessions: dict[str, str],
                       cache_fp: str | Path, max_len: int) -> dict:
    """Extract (or load cached) per-trial reps for every subject's session.

    Returns {worker_id: tensor[T, H]}. Caches to ``cache_fp`` (torch.save).
    """
    import torch

    cache_fp = Path(cache_fp)
    if cache_fp.exists():
        return torch.load(cache_fp, weights_only=False)
    cache_fp.parent.mkdir(parents=True, exist_ok=True)

    reps: dict = {}
    n = len(sessions)
    for i, (wid, text) in enumerate(sessions.items(), 1):
        r = extract_session_reps(model, tokenizer, text, max_len)
        if r is not None:
            reps[wid] = r
        if i % 50 == 0 or i == n:
            print(f"  trial-reps {i}/{n}")
    torch.save(reps, cache_fp)
    print(f"  cached -> {cache_fp}")
    return reps
