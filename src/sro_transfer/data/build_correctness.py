"""Build per-response human-correctness sidecar files alongside the NL data.

For each rendered (source, task) jsonl, produce a parallel `.correct.jsonl`:
    {"worker_id": "s001", "correct": [1, 0, 1, 1, ...]}
where `correct[i]` is the human correctness/reward flag for the trial that
contains the i-th `<<...>>` in the rendered text (replicated across all
responses of a multi-response trial).

Tasks without an objective correct answer (discount-discounting, ...) get
`null` per entry.

Usage:
    python build_correctness.py             # both sources, all tasks
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

# Reuse the renderer's task spec to ensure identical filtering/ordering
import centaur_render as cr

ROOT = Path(__file__).resolve().parent
NL_DIR = ROOT / "output_nl"
RESP_RE = re.compile(r"<<[^>]*>>")


def _correct_value(row, task: str):
    """Per-trial correctness/reward flag, or None if task has no objective answer."""
    # Tasks with no objective correctness — pure preference / discounting / risk.
    # keep_track is included because the released data lacks the word stream, so
    # per-response correctness cannot be reconstructed.
    if task in (
        "kirby", "bickel_titrator", "discount_titrate",
        "columbia_card_task_cold", "columbia_card_task_hot",
        "angling_risk_task_always_sunny", "dietary_decision", "keep_track",
    ):
        return None
    # Tasks where reward feedback is the relevant signal
    if task in ("shift_task", "two_stage_decision"):
        fb = row.get("feedback")
        if pd.notna(fb):
            try:
                return int(fb)
            except (ValueError, TypeError):
                return None
        return None
    # Default: use `correct` column
    c = row.get("correct")
    if pd.isna(c):
        return None
    if isinstance(c, bool):
        return int(c)
    try:
        return int(c)
    except (ValueError, TypeError):
        return None


def build_one_task(source: str, task: str):
    nl_fp = NL_DIR / source / f"{task}.all.jsonl"
    if not nl_fp.exists():
        print(f"  SKIP {source}/{task}: no NL file")
        return
    spec = cr.TASKS[task]
    df = cr.load_task(task, source)

    # Load NL sessions to know <<>> counts per session
    nl_sessions = []
    with nl_fp.open(encoding="utf-8") as f:
        for line in f:
            nl_sessions.append(json.loads(line))

    out_fp = NL_DIR / source / f"{task}.correct.jsonl"
    n_total = 0
    n_nonnull = 0
    with out_fp.open("w", encoding="utf-8") as out:
        for sess in nl_sessions:
            wid = sess["worker_id"]
            sub_df = df[df["worker_id"].astype(str) == wid].sort_values("time_elapsed").reset_index(drop=True)
            # Render each row to get its response count; replicate correct across them
            per_response: list = []
            for _, row in sub_df.iterrows():
                try:
                    trial_text = spec["render"](row)
                except Exception:
                    trial_text = ""
                n_resp = len(RESP_RE.findall(trial_text))
                if n_resp == 0:
                    continue
                c = _correct_value(row, task)
                per_response.extend([c] * n_resp)
            # Sanity: match the count of <<>> in the saved text
            n_in_text = len(RESP_RE.findall(sess["text"]))
            if n_in_text != len(per_response):
                # mismatch — pad/truncate to match (defensive)
                if len(per_response) > n_in_text:
                    per_response = per_response[:n_in_text]
                else:
                    per_response += [None] * (n_in_text - len(per_response))
            out.write(json.dumps({"worker_id": wid, "correct": per_response}) + "\n")
            n_total += len(per_response)
            n_nonnull += sum(1 for c in per_response if c is not None)
    pct_nonnull = 100 * n_nonnull / max(n_total, 1)
    print(f"  {source}/{task:<48} responses={n_total:>7,}  with_correct={pct_nonnull:>5.1f}%  -> {out_fp.name}")


def main():
    for source in ("complete", "retest"):
        if not (NL_DIR / source).exists():
            continue
        for task in cr.TASKS:
            build_one_task(source, task)


if __name__ == "__main__":
    main()
