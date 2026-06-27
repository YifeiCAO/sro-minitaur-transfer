"""In-context cross-task transfer: condition B's predictions on A's full transcript.

No training, no encoder, no injection -- just put the person's entire task-A
session in front of task-B and read off the NLL of their B responses. This is
the FM's native conditioning. The shuffled control (someone else's A) tests
person-specificity.
"""
from __future__ import annotations

import re

from .masking import response_char_spans

_RT = re.compile(r"^rt(\d+|_na)$")               # an RT-bin response token


def _b_label_mask(b_text, tok):
    """Tokenize B (no BOS); per token: is-response flag + is-RT-response flag
    (so we can score choice-only, RT-only, or both)."""
    enc = tok(b_text, add_special_tokens=False, return_offsets_mapping=True)
    spans = response_char_spans(b_text)
    rt_spans = [(rs, re_) for rs, re_ in spans if _RT.match(b_text[rs:re_].strip())]
    flags, rtf = [], []
    for s, e in enc["offset_mapping"]:
        lab = isrt = False
        if s != e:
            for rs, re_ in spans:
                if s < re_ and e > rs:
                    lab = True
                    break
            for rs, re_ in rt_spans:
                if s < re_ and e > rs:
                    isrt = True
                    break
        flags.append(lab); rtf.append(isrt)
    return enc["input_ids"], flags, rtf


def incontext_response_nll(model, tok, b_text, a_text=None, max_len=8192, rep="both"):
    """Mean NLL of B's responses, optionally conditioned on an A-session context.

    A is context only (never scored). If A+B exceeds max_len, tokens are dropped
    from the FRONT (recent A + all of B kept), so the scored B is preserved.
    ``rep`` = "both" | "choice" | "rt" selects which B-response tokens to score.
    """
    import torch
    import torch.nn.functional as F

    b_ids, b_flags, b_rt = _b_label_mask(b_text, tok)
    if a_text is None:
        bos = tok(b_text[:0], add_special_tokens=True)["input_ids"]  # just BOS
        ids = bos + b_ids
        flags = [False] * len(bos) + b_flags
        rtf = [False] * len(bos) + b_rt
    else:
        a_ids = tok(a_text, add_special_tokens=True)["input_ids"]
        ids = a_ids + b_ids
        flags = [False] * len(a_ids) + b_flags
        rtf = [False] * len(a_ids) + b_rt

    if len(ids) > max_len:                      # keep the tail (all B + recent A)
        cut = len(ids) - max_len
        ids, flags, rtf = ids[cut:], flags[cut:], rtf[cut:]

    t = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=t).logits[0]
    sl = logits[:-1].float()
    tl = torch.tensor(ids, device=model.device)[1:]
    mask = torch.tensor(flags, device=model.device, dtype=torch.bool)[1:]
    if rep != "both":
        isrt = torch.tensor(rtf, device=model.device, dtype=torch.bool)[1:]
        mask = (mask & isrt) if rep == "rt" else (mask & ~isrt)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(F.cross_entropy(sl[mask], tl[mask], reduction="mean"))
