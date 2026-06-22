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


def _profile_from_logits(sl, tl):
    """(surprise, entropy) per response from shifted logits ``sl`` [L-1,V] and
    shifted labels ``tl`` [L-1]. Contiguous response positions = one response."""
    import torch
    import torch.nn.functional as F

    mask = tl != -100
    if int(mask.sum()) == 0:
        return None
    pos = mask.nonzero().squeeze(-1)
    logp = F.log_softmax(sl[pos].float(), dim=-1)      # [R, V] only at responses
    toks = tl[pos]
    surp = (-logp[torch.arange(len(pos), device=logp.device), toks]).tolist()
    ent = (-(logp.exp() * logp).sum(-1)).tolist()
    pos_l = pos.tolist()
    rows, i = [], 0
    while i < len(pos_l):
        j = i
        while j + 1 < len(pos_l) and pos_l[j + 1] == pos_l[j] + 1:
            j += 1
        rows.append([float(np.mean(surp[i:j + 1])), float(np.mean(ent[i:j + 1]))])
        i = j + 1
    return np.array(rows, dtype=np.float32)


def extract_surprise_profile(model, tok, text: str, max_len: int):
    """Single-session profile [n_responses, 2] of (mean surprise, mean entropy)."""
    import torch

    e = build_labels(text, tok, max_len)
    ids = torch.tensor([e["input_ids"]], device=model.device)
    att = torch.tensor([e["attention_mask"]], device=model.device)
    lab = torch.tensor(e["labels"], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=att).logits[0]
    return _profile_from_logits(logits[:-1], lab[1:])


def extract_surprise_profiles_batch(model, tok, items, max_len, batch_tokens=8192):
    """Batched profiles for many sessions — fills idle VRAM (e.g. the ~50 GB free
    on a 96 GB card running 70B). ``items`` = list of (wid, text).

    Right-padding + attention_mask: a causal model's logits at real positions are
    identical to the unpadded forward, so batched == per-session (no approximation).
    Sessions are length-bucketed and packed under a B*L ``batch_tokens`` budget.
    """
    import torch

    enc = []
    for w, t in items:
        e = build_labels(t, tok, max_len)
        enc.append((w, e["input_ids"], e["attention_mask"], e["labels"]))
    enc.sort(key=lambda x: len(x[1]))                  # bucket similar lengths
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    out, i, nb = {}, 0, 0
    while i < len(enc):
        L, j = len(enc[i][1]), i                        # ascending -> longest is last added
        while j < len(enc):
            cand_L = max(L, len(enc[j][1]))
            if j > i and cand_L * (j - i + 1) > batch_tokens:
                break
            L = cand_L; j += 1
        batch = enc[i:j]
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        labs = torch.full((len(batch), L), -100, dtype=torch.long)
        for b, (w, iid, am, lb) in enumerate(batch):
            n = len(iid)
            ids[b, :n] = torch.tensor(iid); att[b, :n] = torch.tensor(am); labs[b, :n] = torch.tensor(lb)
        ids = ids.to(model.device); att = att.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits   # [B, L, V]
        for b, (w, *_ ) in enumerate(batch):
            tl = labs[b, 1:].to(model.device)
            p = _profile_from_logits(logits[b, :-1], tl)
            if p is not None:
                out[w] = p
        del logits
        i = j; nb += 1
        if nb % 5 == 0 or i >= len(enc):
            print(f"  surprise {i}/{len(enc)} (batched, {len(batch)} in last batch)")
    return out


def build_or_load_profiles(model, tok, sessions: dict[str, str], cache_fp, max_len,
                           batch_tokens=8192):
    """Cache per-task profiles. ``batch_tokens`` > 0 => batched (fast, fills VRAM);
    set 0 to force the one-at-a-time path."""
    import torch

    cache_fp = Path(cache_fp)
    if cache_fp.exists():
        return torch.load(cache_fp, weights_only=False)   # our own dict-of-numpy cache
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    if batch_tokens and batch_tokens > 0:
        items = list(sessions.items())
        out = extract_surprise_profiles_batch(model, tok, items, max_len, batch_tokens)
        print(f"  surprise {len(out)}/{len(items)} (batched, ~{batch_tokens} tok/batch)")
    else:
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
