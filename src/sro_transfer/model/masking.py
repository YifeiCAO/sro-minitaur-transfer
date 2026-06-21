"""Centaur-style response-token masking.

Loss / likelihood is computed ONLY on the tokens inside ``<<...>>`` (the human's
response). Everything else -- instructions, stimulus descriptions, the ``<<``
and ``>>`` markers -- is context and is masked to -100.

This module is pure-Python + tokenizer; it has no torch dependency so it can be
unit-tested cheaply.
"""
from __future__ import annotations

import re

RESP_SPAN = re.compile(r"<<([^>]*)>>")


def response_char_spans(text: str, include_markers: bool = False) -> list[tuple[int, int]]:
    """Character spans of response content. By default excludes the ``<<``/``>>``."""
    spans = []
    for m in RESP_SPAN.finditer(text):
        if include_markers:
            spans.append((m.start(), m.end()))
        else:
            spans.append((m.start() + 2, m.end() - 2))
    return spans


def build_labels(text: str, tokenizer, max_len: int = 4096):
    """Tokenize ``text`` and return input_ids/attention_mask/labels where labels
    are -100 outside ``<<...>>`` response content.

    Requires a fast tokenizer (offset mapping). Truncates to ``max_len`` tokens.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    spans = response_char_spans(text)
    # O(n) char mask of response positions (was O(tokens x spans) -> very slow on
    # long sessions with many trials). A token is a target if any of its chars is
    # inside a response span.
    resp = bytearray(len(text) + 1)
    for rs, re_ in spans:
        resp[rs:re_] = b"\x01" * (re_ - rs)
    ids = enc["input_ids"]
    labels = [-100] * len(ids)
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s != e and resp.find(b"\x01", s, e) != -1:
            labels[i] = ids[i]
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


def response_token_fraction(text: str, tokenizer, max_len: int = 4096) -> float:
    """Diagnostic: what fraction of tokens are loss-bearing (sanity check)."""
    enc = build_labels(text, tokenizer, max_len)
    n = len(enc["labels"])
    k = sum(1 for x in enc["labels"] if x != -100)
    return k / max(n, 1)
