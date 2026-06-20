"""Per-response SURPRISE profiles under frozen M_pop.

In a choice-only FM the individual signal is the RESIDUAL between a person's
actual choice and the population prediction -- i.e. how surprised M_pop is by
their responses -- NOT the hidden states (which encode the shared, person-
invariant population expectation). For each response we record:
  - surprise = NLL of the human's response token(s) under M_pop
  - entropy  = entropy of M_pop's predicted distribution there (trial difficulty)
A person's profile is the set of (surprise, entropy) over their trials.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .masking import build_labels


def extract_surprise_profile(model, tok, text: str, max_len: int):
    """Return [n_responses, 2] array of (mean surprise, mean entropy) per response."""
    import torch
    import torch.nn.functional as F

    e = build_labels(text, tok, max_len)
    ids = torch.tensor([e["input_ids"]], device=model.device)
    att = torch.tensor([e["attention_mask"]], device=model.device)
    lab = torch.tensor(e["labels"], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=att).logits[0]
    sl, tl = logits[:-1], lab[1:]                      # causal shift
    mask = tl != -100
    if int(mask.sum()) == 0:
        return None
    pos = mask.nonzero().squeeze(-1)
    logp = F.log_softmax(sl[pos].float(), dim=-1)      # [R, V] only at responses
    toks = tl[pos]
    surp = (-logp[torch.arange(len(pos), device=logp.device), toks]).tolist()
    ent = (-(logp.exp() * logp).sum(-1)).tolist()
    pos_l = pos.tolist()
    # group contiguous response positions into one response (mean over subtokens)
    rows, i = [], 0
    while i < len(pos_l):
        j = i
        while j + 1 < len(pos_l) and pos_l[j + 1] == pos_l[j] + 1:
            j += 1
        rows.append([float(np.mean(surp[i:j + 1])), float(np.mean(ent[i:j + 1]))])
        i = j + 1
    return np.array(rows, dtype=np.float32)


def build_or_load_profiles(model, tok, sessions: dict[str, str], cache_fp, max_len):
    import torch

    cache_fp = Path(cache_fp)
    if cache_fp.exists():
        return torch.load(cache_fp)
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    out, n = {}, len(sessions)
    for i, (w, t) in enumerate(sessions.items(), 1):
        p = extract_surprise_profile(model, tok, t, max_len)
        if p is not None:
            out[w] = p
        if i % 50 == 0 or i == n:
            print(f"  surprise {i}/{n}")
    torch.save(out, cache_fp)
    return out


def summarize_profile(profile: np.ndarray) -> np.ndarray:
    """Fixed-length, task-agnostic person vector from a (surprise, entropy) set.

    Captures the distribution of the person's deviations, plus how their surprise
    differs on hard (high-entropy) vs easy trials -- deviation beyond difficulty.
    """
    s, e = profile[:, 0], profile[:, 1]

    def st(x):
        return [float(x.mean()), float(x.std())] + [float(q) for q in np.quantile(x, [.1, .25, .5, .75, .9])]

    feats = st(s) + st(e)
    med = float(np.median(e))
    hi = float(s[e >= med].mean()) if (e >= med).any() else 0.0
    lo = float(s[e < med].mean()) if (e < med).any() else 0.0
    feats += [hi, lo, hi - lo]
    return np.array(feats, dtype=np.float32)            # 17-dim
