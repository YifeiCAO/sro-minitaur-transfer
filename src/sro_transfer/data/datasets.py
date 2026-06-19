"""Load the Centaur-format NL sessions and their correctness sidecars.

Each task has, per source (``complete`` / ``retest``):
  output_nl/<source>/<task>.all.jsonl      -> {worker_id, task, source, text}
  output_nl/<source>/<task>.correct.jsonl  -> {worker_id, correct: [0/1/None,...]}

A "session" is one (subject, task) string in Centaur format: an instruction
paragraph followed by one sentence per trial, with the human's response wrapped
in ``<<...>>``. Loss / prediction targets are exactly those ``<<...>>`` spans.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

RESP_RE = re.compile(r"<<([^>]*)>>")


def _src_dir(nl_dir: str | Path, source: str) -> Path:
    return Path(nl_dir) / source


def available_tasks(nl_dir: str | Path, source: str = "complete") -> list[str]:
    d = _src_dir(nl_dir, source)
    return sorted(p.name[: -len(".all.jsonl")] for p in d.glob("*.all.jsonl"))


def load_sessions(nl_dir: str | Path, task: str, source: str = "complete") -> dict[str, str]:
    """Return {worker_id: session_text} for one task."""
    fp = _src_dir(nl_dir, source) / f"{task}.all.jsonl"
    out: dict[str, str] = {}
    with open(fp, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[str(obj["worker_id"])] = obj["text"]
    return out


def load_correctness(nl_dir: str | Path, task: str, source: str = "complete") -> dict[str, list]:
    """Return {worker_id: [per-response correctness flag]} (None where undefined)."""
    fp = _src_dir(nl_dir, source) / f"{task}.correct.jsonl"
    out: dict[str, list] = {}
    if not fp.exists():
        return out
    with open(fp, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[str(obj["worker_id"])] = obj.get("correct", [])
    return out


def response_tokens(text: str) -> list[str]:
    """The ordered list of human responses (contents of each ``<<...>>``)."""
    return RESP_RE.findall(text)


def iter_responses(
    nl_dir: str | Path, task: str, source: str = "complete"
) -> Iterator[tuple[str, int, str, object]]:
    """Yield (worker_id, response_index, response_token, correct_flag).

    Aligns each ``<<...>>`` with its correctness entry; correctness is None for
    preference/risk tasks and where the sidecar is missing.
    """
    sessions = load_sessions(nl_dir, task, source)
    correct = load_correctness(nl_dir, task, source)
    for wid, text in sessions.items():
        toks = response_tokens(text)
        flags = correct.get(wid, [])
        for i, tok in enumerate(toks):
            yield wid, i, tok, (flags[i] if i < len(flags) else None)
