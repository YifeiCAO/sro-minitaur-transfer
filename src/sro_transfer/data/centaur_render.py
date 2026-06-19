"""Render SRO trial-level data into Psych-101 / Centaur natural-language format.

Each output line of the JSONL = one (subject, task) session, formatted as
declarative prose ("You see X. You press <<Y>>."). Responses are wrapped in
<<...>>, matching the Centaur convention where loss is computed only on those
tokens during fine-tuning.

For each task we provide:
  - INSTR: a short task instruction shown at the top of the session.
  - <task>_trial(row): renders ONE trial as a single English sentence.

Instructions are textbook-accurate descriptions of each task. For three core
tasks (stroop, two_stage_decision, digit_span) we adapted from the actual
expfactory experiment.js text SRO subjects ran; the rest are concise but
faithful task summaries.

Usage:
    python centaur_render.py --source complete --full     # all subjects
    python centaur_render.py --source complete            # 1 demo subject
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOTS = {
    "complete": REPO_ROOT / "Data" / "Complete_02-16-2019" / "Individual_Measures",
    "retest": REPO_ROOT / "Data" / "Retest_02-16-2019" / "Individual_Measures",
}
OUT_DIR = Path(__file__).resolve().parent / "output_nl"
OUT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Helpers
# ===========================================================================

def _parse_list(s) -> list:
    if pd.isna(s):
        return []
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else s
        return list(v)
    except Exception:
        return []


def _parse_int_list(s) -> list[int]:
    return [int(x) for x in _parse_list(s)]


def _safe_int(x, default=None):
    if pd.isna(x):
        return default
    try:
        return int(x)
    except Exception:
        return default


def _arrow(kp) -> str:
    """37/38/39/40 → left/up/right/down. Returns <<left>> etc or (no response)."""
    if pd.isna(kp) or kp == -1:
        return "(no response)"
    m = {37: "left", 38: "up", 39: "right", 40: "down"}
    label = m.get(int(kp))
    return f"<<{label}>>" if label else "<<?>>"


def _letter_key(kp) -> str:
    """65-90 → a-z."""
    if pd.isna(kp) or kp == -1:
        return "(no response)"
    code = int(kp)
    if 65 <= code <= 90:
        return f"<<{chr(code).lower()}>>"
    return "<<?>>"


# ===========================================================================
# stroop
# ===========================================================================
STROOP_INSTR = (
    "In this experiment you will see color words (RED, BLUE, GREEN) appear one "
    "at a time. The ink of the words will also be colored. Your task is to "
    "press the key corresponding to the ink color of the word, ignoring the "
    "word's meaning. Press 'r' for words colored red, 'b' for words colored "
    "blue, and 'g' for words colored green."
)
STROOP_KEY = {66: "b", 71: "g", 82: "r"}


def stroop_trial(row):
    word = str(row["stim_word"]).upper()
    ink = str(row["stim_color"]).lower()
    kp = row.get("key_press")
    resp = (
        "(no response)"
        if pd.isna(kp) or kp == -1
        else f"<<{STROOP_KEY.get(int(kp), '?')}>>"
    )
    return f"The word {word} appears in {ink} ink. You press {resp}."


# ===========================================================================
# simon
# ===========================================================================
SIMON_INSTR = (
    "On each trial, a colored square (red or blue) appears on the left or right "
    "side of the screen. Your task is to report the COLOR of the square by "
    "pressing the left or right arrow key, ignoring the position. The "
    "color-to-key mapping is consistent throughout your session."
)


def simon_trial(row):
    color = row["stim_color"]
    side = row["stim_side"]
    return f"A {color} square appears on the {side}. You press {_arrow(row.get('key_press'))}."


# ===========================================================================
# attention_network_task (ANT)
# ===========================================================================
ANT_INSTR = (
    "On each trial, a row of five arrows appears above or below a central "
    "fixation cross. Sometimes a warning cue appears beforehand to direct your "
    "attention. Your task is to report the direction of the CENTER arrow, "
    "ignoring the surrounding flanker arrows. Press the left arrow key if the "
    "center points left, the right arrow key if it points right."
)
ANT_CUE = {
    "spatial": "A spatial cue appears at the upcoming target location.",
    "nocue": "No warning cue appears.",
    "double": "Two warning cues appear, one above and one below fixation.",
    "center": "A warning cue appears at the center.",
}


def ant_trial(row):
    cue_txt = ANT_CUE.get(row.get("cue"), f"A {row.get('cue')} cue appears.")
    loc = row.get("flanker_location", "?")
    target = row.get("flanker_middle_direction", "?")
    fl_type = row.get("flanker_type", "?")
    return (
        f"{cue_txt} A row of arrows then appears {loc} the fixation: the "
        f"center arrow points {target}, with {fl_type} flanking arrows. You "
        f"press {_arrow(row.get('key_press'))}."
    )


# ===========================================================================
# directed_forgetting
# ===========================================================================
DF_INSTR = (
    "On each trial, you first see two sets of three letters, one above the "
    "other. After a brief delay, a cue tells you which set to REMEMBER and "
    "which to forget. Then a single probe letter appears, and you report "
    "whether the probe was in the set you were told to remember. Press the "
    "left arrow key for 'yes' (probe was in the remembered set), the right "
    "arrow key for 'no'."
)


def _lstr(lst):
    return " ".join(str(x) for x in lst)


def df_trial(row):
    top = _parse_list(row.get("stim_top"))
    bot = _parse_list(row.get("stim_bottom"))
    cue = str(row.get("cue", "")).lower()
    cue_word = "top" if cue == "top" else "bottom"
    probe = row.get("probe_letter", "?")
    kp = row.get("key_press")
    if pd.isna(kp) or kp == -1:
        resp = "(no response)"
    elif int(kp) == 37:
        resp = "<<yes>>"
    elif int(kp) == 39:
        resp = "<<no>>"
    else:
        resp = "<<?>>"
    return (
        f"The top set is [{_lstr(top)}] and the bottom set is [{_lstr(bot)}]. "
        f"You are cued to remember the {cue_word} set. The probe letter is "
        f"{probe}. You press {resp}."
    )


# ===========================================================================
# recent_probes
# ===========================================================================
RP_INSTR = (
    "On each trial, you see a set of six letters to remember. After a delay, a "
    "probe letter appears and you report whether it was in the just-shown set. "
    "On some trials, the probe was in a recent earlier set but NOT the current "
    "one — these 'recent lures' are easy to confuse. Press the left arrow key "
    "for 'yes' (probe is in the current set), the right arrow key for 'no'."
)


def rp_trial(row):
    cur = _parse_list(row.get("stim"))
    probe = row.get("probe_letter", "?")
    kp = row.get("key_press")
    if pd.isna(kp) or kp == -1:
        resp = "(no response)"
    elif int(kp) == 37:
        resp = "<<yes>>"
    elif int(kp) == 39:
        resp = "<<no>>"
    else:
        resp = "<<?>>"
    return (
        f"The current letter set is [{_lstr(cur)}]. The probe letter is "
        f"{probe}. You press {resp}."
    )


# ===========================================================================
# dot_pattern_expectancy (DPX / AX-CPT)
# ===========================================================================
DPX_INSTR = (
    "On each trial, a cue pattern (A or B) appears followed by a probe pattern "
    "(X or Y). You must press the TARGET key only when the cue is A and the "
    "probe is X (an 'AX' trial), and the NONTARGET key for all other "
    "combinations (AY, BX, BY). Press the left arrow key for target, the down "
    "arrow key for nontarget."
)


def dpx_trial(row):
    cond = row.get("condition", "??")
    cue, probe = cond[0], cond[1]
    kp = row.get("key_press")
    if pd.isna(kp) or kp == -1:
        resp = "(no response)"
    elif int(kp) == 37:
        resp = "<<target>>"
    elif int(kp) == 40:
        resp = "<<nontarget>>"
    else:
        resp = "<<?>>"
    return f"Cue {cue} appears, followed by probe {probe}. You press {resp}."


# ===========================================================================
# choice_reaction_time
# ===========================================================================
CRT_INSTR = (
    "On each trial, one of two visual stimuli appears at the center of the "
    "screen. You report which stimulus appeared by pressing one of two keys "
    "(the stimulus-to-key mapping is consistent throughout your session)."
)


def crt_trial(row):
    sid = _safe_int(row.get("stim_id"), 0)
    return f"Stimulus {sid} appears. You press {_letter_key(row.get('key_press'))}."


# ===========================================================================
# threebytwo
# ===========================================================================
T32_INSTR = (
    "On each trial, a number from 1 to 9 (excluding 5) appears in either orange "
    "or blue ink, preceded by a task cue. The cue tells you which judgment to "
    "make about the number: 'Parity' (or 'Odd-Even') = is the number odd or "
    "even? 'Magnitude' (or 'High-Low') = is the number greater or less than 5? "
    "'Color' (or 'Orange-Blue') = is the ink orange or blue? Press one of two "
    "keys to indicate your answer."
)


def t32_trial(row):
    cue = str(row.get("cue", "?"))
    color = row.get("stim_color", "?")
    n = _safe_int(row.get("stim_number"), 0)
    return (
        f"The cue '{cue}' appears, then the number {n} appears in {color} ink. "
        f"You press {_letter_key(row.get('key_press'))}."
    )


# ===========================================================================
# shape_matching
# ===========================================================================
SM_INSTR = (
    "On each trial, a probe shape appears at the top of the screen. After a "
    "brief delay, two more shapes (a target and a distractor) appear below. "
    "You must indicate whether the PROBE matches the TARGET shape. Press one "
    "key for 'match', another for 'no match'."
)


def sm_trial(row):
    p = _safe_int(row.get("probe_id"), 0)
    t = _safe_int(row.get("target_id"), 0)
    d = _safe_int(row.get("distractor_id"), 0)
    kp = row.get("key_press")
    if pd.isna(kp) or kp == -1:
        resp = "(no response)"
    elif int(kp) == 77:
        resp = "<<match>>"
    elif int(kp) == 90:
        resp = "<<nomatch>>"
    else:
        resp = "<<?>>"
    return (
        f"The probe is shape {p}. The target is shape {t} and the distractor "
        f"is shape {d}. You press {resp}."
    )


# ===========================================================================
# local_global_letter (Navon)
# ===========================================================================
LGL_INSTR = (
    "On each trial, you see a large letter (the 'global' letter) made up of "
    "many smaller letters (the 'local' letters). A cue tells you whether to "
    "report the GLOBAL letter or the LOCAL letter. Press H for the letter H, "
    "S for the letter S."
)


def lgl_trial(row):
    g = row.get("global_shape", "?")
    l = row.get("local_shape", "?")
    cond = row.get("condition", "?")
    kp = row.get("key_press")
    if pd.isna(kp) or kp == -1:
        resp = "(no response)"
    elif int(kp) == 72:
        resp = "<<H>>"
    elif int(kp) == 83:
        resp = "<<S>>"
    else:
        resp = "<<?>>"
    return (
        f"A large {g.upper()} made of small {l.upper()}s appears. You are "
        f"cued to report the {cond} letter. You press {resp}."
    )


# ===========================================================================
# stop_signal family (3 variants share template)
# ===========================================================================
def _ss_instr(extra: str = "") -> str:
    base = (
        "On each trial, a shape appears and you must respond as quickly as "
        "possible with one of two keys depending on the shape. On some trials, "
        "a stop signal appears shortly after the shape — when this happens, "
        "you must NOT respond. The stop-signal delay varies from trial to "
        "trial."
    )
    return base + (" " + extra if extra else "")


STOP_SIGNAL_INSTR = _ss_instr()
MOTOR_SS_INSTR = _ss_instr(
    "In this variant, the stop signal applies only to one of the two go-shapes "
    "('stop' condition); for the other shape ('ignore' condition) you should "
    "respond even when a stop signal appears."
)
STIM_SS_INSTR = _ss_instr(
    "In this variant, the stop signal applies only to one specific stimulus "
    "color/identity; for the other you should respond even when a stop signal "
    "appears."
)


def _ss_trial(row, has_condition_label=False):
    cr = _safe_int(row.get("correct_response"))
    go_key = _letter_key(cr).strip("<>") if cr is not None else "?"  # the key the shape calls for
    ss_type = str(row.get("SS_trial_type", "go"))
    ssd = _safe_int(row.get("SS_delay"), 0)
    cond = str(row.get("condition", "")) if has_condition_label else ""
    cond_str = f" ({cond} condition)" if cond and cond not in ("high", "low") else ""

    kp = row.get("key_press")
    resp = _letter_key(kp)

    if ss_type == "stop":
        stim_part = f"A go-shape (calling for key '{go_key}') appears{cond_str}, then a STOP signal appears after {ssd} ms"
    else:
        stim_part = f"A go-shape (calling for key '{go_key}') appears{cond_str}, no stop signal"
    return f"{stim_part}. You press {resp}."


def stop_signal_trial(row):
    return _ss_trial(row, has_condition_label=False)


def motor_ss_trial(row):
    return _ss_trial(row, has_condition_label=True)


def stim_ss_trial(row):
    return _ss_trial(row, has_condition_label=True)


# ===========================================================================
# go_nogo
# ===========================================================================
GNG_INSTR = (
    "On each trial, a shape appears. On 'go' trials, press the space bar as "
    "quickly as possible. On 'no-go' trials, do NOT respond."
)


def gng_trial(row):
    cond = str(row.get("condition", "?"))
    kp = row.get("key_press")
    if pd.isna(kp) or kp == -1:
        resp = "(no response)"
    elif int(kp) == 32:
        resp = "<<space>>"
    else:
        resp = "<<?>>"
    return f"A {cond} signal appears. You press {resp}."


# ===========================================================================
# hierarchical_rule
# ===========================================================================
HR_INSTR = (
    "On each trial, an image appears surrounded by a colored border with a "
    "particular orientation. The combination of image, border color, and "
    "orientation determines which of three keys (J, K, or L) you should press, "
    "following a hierarchical rule that you must learn."
)


def hr_trial(row):
    border = row.get("border", "?")
    orient = row.get("orientation", "?")
    s = _safe_int(row.get("stim"), 0)
    return (
        f"Image {s} appears with a {border} border in {orient} orientation. "
        f"You press {_letter_key(row.get('key_press'))}."
    )


# ===========================================================================
# adaptive_n_back
# ===========================================================================
NBACK_INSTR = (
    "On each trial, a letter appears. You must report whether the current "
    "letter matches the letter that appeared N trials ago, where N is the "
    "current 'load' level (which adapts based on your performance). Press the "
    "left arrow key for 'match' (target), and make no response for 'no match' "
    "(nontarget)."
)


def nback_trial(row):
    stim = str(row.get("stim", "?")).lower()
    load = _safe_int(row.get("load"), 0)
    target = row.get("target")
    # Early trials of a block may not have an n-back target yet
    if pd.isna(target) or target == "" or str(target).lower() == "nan":
        target_str = ""
    else:
        target_str = f", n-back target was '{str(target).lower()}'"
    kp = row.get("key_press")
    # n-back convention: any key press = "match" response; no press = nontarget
    if pd.isna(kp) or kp == -1:
        resp = "<<nontarget>>"
    else:
        resp = "<<target>>"
    return (
        f"The letter '{stim}' appears (load = {load}{target_str}). "
        f"You press {resp}."
    )


# ===========================================================================
# probabilistic_selection
# ===========================================================================
PS_INSTR = (
    "On each trial during this test phase, two abstract images appear side by "
    "side. From the earlier training phase, you have learned the reward "
    "probabilities associated with each image. You now choose the image you "
    "believe is better, with no feedback. Press the left arrow key for the "
    "left image, the right arrow key for the right image."
)
_PS_HTML_RE = re.compile(r"decision-(left|right)><img src='[^']*?/(\d+)\.png'")


def ps_trial(row):
    stim = row.get("stimulus", "") or ""
    left, right = None, None
    for side, num in _PS_HTML_RE.findall(stim):
        if side == "left":
            left = int(num)
        else:
            right = int(num)
    left_s = left if left is not None else "?"
    right_s = right if right is not None else "?"
    kp = row.get("key_press")
    if pd.isna(kp) or kp == -1:
        resp = "(no response)"
    elif int(kp) == 37:
        resp = "<<left>>"
    elif int(kp) == 39:
        resp = "<<right>>"
    else:
        resp = "<<?>>"
    return f"Image {left_s} on the left, image {right_s} on the right. You press {resp}."


# ===========================================================================
# delay-discounting family (kirby, bickel_titrator, discount_titrate)
# ===========================================================================
DD_INSTR = (
    "On each trial, you choose between a smaller monetary reward available "
    "sooner and a larger monetary reward available later. There are no right "
    "or wrong answers; just choose the option you prefer."
)


def _choice_resp(row):
    c = row.get("choice")
    if pd.isna(c):
        return "(no response)"
    c = str(c)
    if c in ("smaller_sooner", "larger_later"):
        return f"<<{c}>>"
    return "<<?>>"


def kirby_trial(row):
    ss = _safe_int(row.get("small_amount"), 0)
    ll = _safe_int(row.get("large_amount"), 0)
    delay = _safe_int(row.get("later_delay"), 0)
    return (
        f"You choose between ${ss} today and ${ll} in {delay} days. "
        f"You press {_choice_resp(row)}."
    )


def bickel_trial(row):
    ss = _safe_int(row.get("smaller_amount"), 0)
    ll = _safe_int(row.get("larger_amount"), 0)
    sooner = _safe_int(row.get("sooner_time_days"), 0)
    later = _safe_int(row.get("later_time_days"), 0)
    sooner_str = "today" if sooner == 0 else f"in {sooner} days"
    later_str = f"in {later} days"
    return (
        f"You choose between ${ss} {sooner_str} and ${ll} {later_str}. "
        f"You press {_choice_resp(row)}."
    )


def discount_trial(row):
    ss = row.get("smaller_amount")
    ll = row.get("larger_amount")
    sooner = _safe_int(row.get("sooner_days"), 0)
    later = _safe_int(row.get("later_days"), 0)
    ss_s = f"{ss:g}" if pd.notna(ss) else "?"
    ll_s = f"{ll:g}" if pd.notna(ll) else "?"
    sooner_str = "today" if sooner == 0 else f"in {sooner} days"
    later_str = f"in {later} days"
    return (
        f"You choose between ${ss_s} {sooner_str} and ${ll_s} {later_str}. "
        f"You press {_choice_resp(row)}."
    )


# ===========================================================================
# psychological_refractory_period_two_choices (PRP)
# ===========================================================================
PRP_INSTR = (
    "On each trial, two stimuli appear in quick succession. The first is a "
    "colored patch (blue or yellow) and the second is a single number (3 or "
    "4). You must respond to BOTH stimuli in order: first to the color, then "
    "to the number, as fast as possible. The interval between the two stimuli "
    "varies from trial to trial."
)


def prp_trial(row):
    s1 = row.get("choice1_stim")
    s2 = row.get("choice2_stim")
    isi = _safe_int(row.get("ISI"), 0)
    if pd.isna(s1):
        s1_desc = "(no stimulus)"
    else:
        s1_desc = f"a {s1} patch"
    if pd.isna(s2):
        s2_desc = "(no stimulus)"
    else:
        try:
            s2_desc = f"the number {int(s2) if float(s2).is_integer() else s2}"
        except Exception:
            s2_desc = f"the symbol {s2}"
    kp1 = row.get("choice1_key_press")
    kp2 = row.get("choice2_key_press")
    return (
        f"First, {s1_desc} appears. After {isi} ms, the second stimulus appears: "
        f"{s2_desc}. You press {_letter_key(kp1)}, then {_letter_key(kp2)}."
    )


# ===========================================================================
# ravens
# ===========================================================================
RAVENS_INSTR = (
    "On each trial, you see a Raven's Progressive Matrices puzzle: a grid of "
    "patterns with one piece missing. Below are eight candidate pieces, "
    "labeled A through H. Choose the piece that best completes the pattern."
)
_RAVENS_RE = re.compile(r"images/(?:practice/)?(?:top|bottom)_(\d+)")


def ravens_trial(row):
    m = _RAVENS_RE.search(str(row.get("stim_question", "")))
    pid = m.group(1) if m else "?"
    r = row.get("stim_response")
    if pd.isna(r) or r == "":
        resp = "(no response)"
    else:
        resp = f"<<{str(r).upper()}>>"
    return f"Puzzle {pid} appears with options A through H. You select {resp}."


# ===========================================================================
# shift_task
# ===========================================================================
SHIFT_INSTR = (
    "On each trial, three abstract shapes appear, each varying in shape, "
    "color, and pattern. Choose one shape; you will receive feedback whether "
    "your choice was correct. Which feature determines correctness must be "
    "learned by trial and error, and the rewarded feature shifts "
    "unpredictably during the experiment."
)


def shift_trial(row):
    try:
        stims = ast.literal_eval(row["stims"]) if isinstance(row.get("stims"), str) else []
    except Exception:
        stims = []
    desc_parts = []
    for i, s in enumerate(stims, start=1):
        if isinstance(s, dict):
            desc_parts.append(
                f"Option {i} is a {s.get('color','?')} {s.get('shape','?')} "
                f"with {s.get('pattern','?')} pattern"
            )
    desc = ". ".join(desc_parts) + "." if desc_parts else "Three shapes appear."
    pos = _safe_int(row.get("choice_position"))
    if pos is None or pos not in (1, 2, 3):
        resp = "(no response)"
    else:
        resp = f"<<{pos}>>"
    fb = row.get("feedback")
    if pd.isna(fb):
        fb_str = ""
    elif int(fb) == 1:
        fb_str = " Correct."
    else:
        fb_str = " Incorrect."
    return f"{desc} You press {resp}.{fb_str}"


# ===========================================================================
# spatial_span
# ===========================================================================
SS_INSTR = (
    "On each trial, a sequence of squares on a grid lights up one at a time. "
    "After the sequence completes, click the squares in the order they lit "
    "up. For reverse trials, click them in the opposite order."
)
SS_INSTR_REVERSE_NOTE = (
    " For the reverse trials, enter the sequence backwards: the last position "
    "first, the second-to-last second, and so on."
)


def spatial_span_trial(row):
    direction = row.get("condition", "forward")
    shown = _parse_int_list(row.get("sequence"))
    resp = _parse_int_list(row.get("response"))
    shown_str = " ".join(str(p) for p in shown) if shown else "(none)"
    if not resp:
        resp_part = "You enter (no response)."
    else:
        parts = [f"<<{p}>>" for p in resp]
        resp_part = "You click " + parts[0] + "".join(f", then {p}" for p in parts[1:]) + "."
    return f"A {direction} sequence of grid positions lights up: {shown_str}. {resp_part}"


# ===========================================================================
# digit_span (kept here for consistency)
# ===========================================================================
DIGIT_INSTR = (
    "In this test you will see a sequence of numbers appear on the screen one "
    "after the other. At the end of each trial, enter all the numbers in the "
    "order in which they appeared."
)
DIGIT_INSTR_REVERSE_NOTE = (
    " For the reverse trials, enter the sequence backwards: the last item "
    "first, the second-to-last second, and so on."
)


def digit_trial(row):
    direction = row.get("condition", "forward")
    shown = _parse_int_list(row.get("sequence"))
    resp = _parse_int_list(row.get("response"))
    shown_str = " ".join(str(d) for d in shown) if shown else "(none)"
    if not resp:
        resp_part = "You enter (no response)."
    else:
        parts = [f"<<{d}>>" for d in resp]
        resp_part = "You enter " + parts[0] + "".join(f", then {p}" for p in parts[1:]) + "."
    return f"A {direction} sequence is shown: {shown_str}. {resp_part}"


# ===========================================================================
# two_stage_decision
# ===========================================================================
TWOSTAGE_INSTR = (
    "In this task, you make decisions in two stages to get a reward. In the "
    "first stage, two abstract shapes appear on colored backgrounds and you "
    "choose one. Each first-stage choice is primarily associated with one of "
    "two second-stages (the common transition is more frequent than the rare "
    "transition, and these transition probabilities are fixed throughout). "
    "In the second stage, two more shapes appear and you choose one. Each "
    "second-stage shape has its own probability of paying a gold coin, and "
    "these reward probabilities slowly drift over the course of the "
    "experiment. Try to earn as many coins as possible. Press the left arrow "
    "key to choose the left shape and the right arrow key to choose the right "
    "shape."
)
S2_PAIR = {1: (2, 3), 2: (4, 5)}


def twostage_trial(row):
    try:
        order = ast.literal_eval(row["stim_order_first"])
        s_left, s_right = int(order[0]), int(order[1])
    except Exception:
        s_left, s_right = 0, 1
    kp1 = row.get("key_press_first")
    if pd.isna(kp1) or kp1 == -1:
        resp1 = "(no response)"
    elif int(kp1) == 37:
        resp1 = "<<left>>"
    elif int(kp1) == 39:
        resp1 = "<<right>>"
    else:
        resp1 = "<<?>>"

    s2 = row.get("stage_second")
    if pd.notna(s2) and int(s2) in S2_PAIR:
        a, b = S2_PAIR[int(s2)]
        stage_name = "second-stage A" if int(s2) == 1 else "second-stage B"
        transition = row.get("stage_transition", "")
        if isinstance(transition, str) and transition:
            article = "an" if transition[0] in "aeiou" else "a"
            trans_str = f" ({article} {transition} transition)"
        else:
            trans_str = ""
    else:
        a, b = "?", "?"
        stage_name = "the next stage"
        trans_str = ""

    kp2 = row.get("key_press_second")
    if pd.isna(kp2) or kp2 == -1:
        resp2 = "(no response)"
    elif int(kp2) == 37:
        resp2 = "<<left>>"
    elif int(kp2) == 39:
        resp2 = "<<right>>"
    else:
        resp2 = "<<?>>"

    fb = row.get("feedback")
    if pd.isna(fb):
        fb_str = ""
    elif int(fb) == 1:
        fb_str = " You receive a gold coin."
    else:
        fb_str = " You receive no coin."

    return (
        f"In stage 1, shape {s_left} is on the left and shape {s_right} is on the right. "
        f"You press {resp1}. This takes you to {stage_name}{trans_str}, where "
        f"shape {a} is on the left and shape {b} is on the right. You press {resp2}.{fb_str}"
    )


# ===========================================================================
# columbia_card_task_cold (deliberative risk; one decision per round)
# ===========================================================================
CCT_COLD_INSTR = (
    "On each round, you see a grid of 32 face-down cards. A fixed number of them "
    "are loss cards; the rest are gain cards. Each gain card you turn is worth a "
    "small reward, but turning a single loss card ends the round and subtracts a "
    "large penalty. You are told, before deciding, how many loss cards there are, "
    "how much each gain card is worth, and how much a loss card costs. You then "
    "state how many cards you want to turn over — all at once, without feedback. "
    "Choosing more cards means more potential gain but more risk."
)


def cct_cold_trial(row):
    gain = _safe_int(row.get("gain_amount"), 0)
    loss = _safe_int(row.get("loss_amount"), 0)
    nloss = _safe_int(row.get("num_loss_cards"), 0)
    k = _safe_int(row.get("num_cards_chosen"))
    resp = "(no response)" if k is None else f"<<{k}>>"
    return (
        f"This round: {nloss} of the 32 cards are loss cards. Each gain card is "
        f"worth +{gain} points; a loss card costs {loss} points and ends the "
        f"round. You choose to turn over {resp} cards."
    )


# ===========================================================================
# columbia_card_task_hot (affective/sequential risk; turn cards one at a time)
# ===========================================================================
CCT_HOT_INSTR = (
    "On each round, you see a grid of face-down cards. A fixed number of them are "
    "loss cards; the rest are gain cards. You turn the cards over one at a time. "
    "Each gain card adds a small reward and you see your running total grow; you "
    "may stop and bank your points at any moment. But if you turn over a loss "
    "card, the round ends and you lose a large penalty. You are told how many "
    "loss cards there are, how much each gain card is worth, and the penalty. "
    "Decide how far to push your luck before collecting."
)


def cct_hot_trial(row):
    gain = _safe_int(row.get("gain_amount"), 0)
    loss = _safe_int(row.get("loss_amount"), 0)
    nloss = _safe_int(row.get("num_loss_cards"), 0)
    n = _safe_int(row.get("n_cards_turned"), 0)
    busted = int(row.get("busted", 0) or 0)
    pts = _safe_int(row.get("round_points"), 0)
    head = (
        f"This round: {nloss} loss cards; each gain card is worth +{gain} points, "
        f"a loss card costs {loss} points."
    )
    if busted:
        return (
            f"{head} You turn over <<{n}>> cards, then hit a loss card and lose "
            f"the round ({pts} points)."
        )
    return (
        f"{head} You turn over <<{n}>> cards, then stop and bank {pts} points."
    )


# ===========================================================================
# angling_risk_task_always_sunny (sequential risk; keep fishing vs collect)
# ===========================================================================
ART_INSTR = (
    "You go on a series of fishing trips at a lake. On each cast you reel in a "
    "fish: a blue fish adds one to your trip tank, but a red fish ends the trip "
    "and you lose everything in the tank. After each blue fish you may keep "
    "fishing or stop and bank your tank into your tournament total. (The weather "
    "is always sunny, so the only risk is the red fish.) The more fish already "
    "in the lake you have caught, the riskier each further cast becomes."
)


def art_trial(row):
    tid = str(row.get("trial_id", ""))
    tank = _safe_int(row.get("trip_bank"), 0)
    if tid == "stim":
        return (
            f"Your tank holds {tank} blue fish. You keep fishing and reel in "
            f"another blue fish. You press <<keep>>."
        )
    if tid == "round_over":
        busted = _safe_int(row.get("caught_blue"), 0) == 1
        if busted:
            return (
                f"Your tank holds {tank} blue fish. You keep fishing, but reel in "
                f"a red fish — the trip ends and you lose your tank. You press "
                f"<<keep>>."
            )
        return (
            f"Your tank holds {tank} blue fish. You stop and bank them into your "
            f"tournament total. You press <<collect>>."
        )
    return f"(skipped {tid})"


# ===========================================================================
# information_sampling_task (reflection-impulsivity; sample then decide)
# ===========================================================================
IST_INSTR = (
    "On each round, you see a grid of covered boxes. You open them one at a time, "
    "and each reveals one of several colors. Your goal is to decide which color "
    "is in the majority across all the boxes. You may open as few or as many "
    "boxes as you like before committing to a guess. In 'fixed win' rounds the "
    "reward for a correct guess is constant; in 'decreasing win' rounds the "
    "reward shrinks with every box you open, so sampling has a cost."
)


def ist_trial(row):
    tid = str(row.get("trial_id", ""))
    color = row.get("color_clicked")
    if tid == "stim":
        c = str(color) if pd.notna(color) else "a color"
        return f"You open a box; it reveals {c}."
    if tid == "choice":
        if pd.isna(color):
            resp = "(no response)"
        else:
            resp = f"<<{str(color)}>>"
        return f"You stop opening boxes and judge the majority color to be {resp}."
    return f"(skipped {tid})"


# ===========================================================================
# dietary_decision (self-control; eat or not, health vs taste)
# ===========================================================================
DIET_INSTR = (
    "Earlier you rated a set of foods on how healthy and how tasty each one is. "
    "Now, on each trial, a single food appears and you decide whether you would "
    "like to eat it at the end of the experiment instead of a neutral reference "
    "snack. You answer on a five-point scale from strong no to strong yes. "
    "Exercising dietary self-control means weighting health over taste when they "
    "conflict."
)
_DIET_RESP = {2: "strong_yes", 1: "yes", 0: "neutral", -1: "no", -2: "strong_no", -3: "strong_no"}


def _diff_phrase(diff, more, less, same):
    d = _safe_int(diff)
    if d is None:
        return None
    if d > 0:
        return more
    if d < 0:
        return less
    return same


def diet_trial(row):
    food = str(row.get("stim", "?"))
    cr = row.get("coded_response")
    if pd.isna(cr):
        resp = "(no response)"
    else:
        resp = f"<<{_DIET_RESP.get(int(cr), 'neutral')}>>"
    h = _diff_phrase(row.get("health_diff"), "healthier", "less healthy", "similarly healthy")
    t = _diff_phrase(row.get("taste_diff"), "tastier", "less tasty", "similarly tasty")
    if h and t:
        rated = f" You had rated it {h} and {t} than your reference snack."
    else:
        rated = ""
    return f"The food '{food}' appears.{rated} You respond {resp}."


# ===========================================================================
# tower_of_london (planning; move balls peg to peg to match a goal)
# ===========================================================================
TOL_INSTR = (
    "On each problem you see three pegs holding colored balls in a starting "
    "arrangement, and a goal arrangement to reach. You move one ball at a time "
    "from the top of one peg to another, and the pegs hold different maximum "
    "numbers of balls. Reach the goal in as few moves as possible; plan your "
    "moves before you start."
)
_TOL_PEG_RE = re.compile(r"tol_peg_(\d+)")


def _tol_board(state_str):
    """Render '[[1,0,0],[3,0],[2]]' as a readable per-peg description."""
    try:
        pegs = ast.literal_eval(state_str)
    except Exception:
        return "?"
    parts = []
    for i, peg in enumerate(pegs, start=1):
        balls = [str(b) for b in peg if b not in (0, "0")]
        if balls:
            parts.append(f"peg {i} holds ball(s) {' '.join(balls)}")
        else:
            parts.append(f"peg {i} is empty")
    return "; ".join(parts)


def _tol_peg(mouse_click):
    m = _TOL_PEG_RE.search(str(mouse_click))
    return m.group(1) if m else "?"


def tol_trial(row):
    tid = str(row.get("trial_id", ""))
    if tid == "feedback":
        c = row.get("correct")
        if pd.isna(c):
            return "The problem ends."
        return "The problem ends. " + ("You solved it." if int(c) == 1 else "You did not solve it in time.")
    peg = _tol_peg(row.get("mouse_click"))
    if tid == "to_hand":
        board = _tol_board(row.get("current_position"))
        target = _tol_board(row.get("target"))
        return (
            f"Current arrangement: {board}. Goal: {target}. You lift the top ball "
            f"off peg {peg}. You press <<peg_{peg}>>."
        )
    if tid == "to_board":
        return f"You drop the ball on peg {peg}. You press <<peg_{peg}>>."
    return f"(skipped {tid})"


# ===========================================================================
# keep_track (working-memory updating; report last word per category)
# NOTE: the scrolling word stream is NOT in the released data — only the target
# categories and the subject's final recall are available, so the recalled
# words are not predictable from the input. Encoded for completeness; the only
# individual-difference signal here is recall span (how many categories filled).
# ===========================================================================
KT_INSTR = (
    "On each trial you are given several target categories (for example colors, "
    "animals, metals). A long stream of words then scrolls by, and you must keep "
    "track of, and remember, the most recent word you saw belonging to each "
    "target category. At the end you report the last word for each category."
)


def kt_trial(row):
    cats = _parse_list(row.get("targets"))
    cats_str = ", ".join(str(c) for c in cats) if cats else "(none)"
    resp = _parse_list(row.get("responses"))
    if not resp:
        recall = "(no response)"
    else:
        parts = [f"<<{str(w).strip()}>>" for w in resp if str(w).strip()]
        recall = ", ".join(parts) if parts else "(no response)"
    return (
        f"You must track the most recent word in each of these categories: "
        f"{cats_str}. After the word stream ends, you recall: {recall}."
    )


# ===========================================================================
# Task registry
# ===========================================================================
TASKS: dict[str, dict] = {
    "stroop":                          {"instr": STROOP_INSTR,         "render": stroop_trial,         "extra_stages": ()},
    "simon":                           {"instr": SIMON_INSTR,          "render": simon_trial,          "extra_stages": ()},
    "attention_network_task":          {"instr": ANT_INSTR,            "render": ant_trial,            "extra_stages": ()},
    "directed_forgetting":             {"instr": DF_INSTR,             "render": df_trial,             "extra_stages": ()},
    "recent_probes":                   {"instr": RP_INSTR,             "render": rp_trial,             "extra_stages": ()},
    "dot_pattern_expectancy":          {"instr": DPX_INSTR,            "render": dpx_trial,            "extra_stages": ()},
    "choice_reaction_time":            {"instr": CRT_INSTR,            "render": crt_trial,            "extra_stages": ()},
    "threebytwo":                      {"instr": T32_INSTR,            "render": t32_trial,            "extra_stages": ()},
    "shape_matching":                  {"instr": SM_INSTR,             "render": sm_trial,             "extra_stages": ()},
    "local_global_letter":             {"instr": LGL_INSTR,            "render": lgl_trial,            "extra_stages": ()},
    "stop_signal":                     {"instr": STOP_SIGNAL_INSTR,    "render": stop_signal_trial,    "extra_stages": ()},
    "motor_selective_stop_signal":     {"instr": MOTOR_SS_INSTR,       "render": motor_ss_trial,       "extra_stages": ()},
    "stim_selective_stop_signal":      {"instr": STIM_SS_INSTR,        "render": stim_ss_trial,        "extra_stages": ()},
    "go_nogo":                         {"instr": GNG_INSTR,            "render": gng_trial,            "extra_stages": ()},
    "hierarchical_rule":               {"instr": HR_INSTR,             "render": hr_trial,             "extra_stages": ()},
    "adaptive_n_back":                 {"instr": NBACK_INSTR,          "render": nback_trial,          "extra_stages": ("adaptive", "control")},
    "probabilistic_selection":         {"instr": PS_INSTR,             "render": ps_trial,             "extra_stages": ()},
    "kirby":                           {"instr": DD_INSTR,             "render": kirby_trial,          "extra_stages": ()},
    "bickel_titrator":                 {"instr": DD_INSTR,             "render": bickel_trial,         "extra_stages": ()},
    "discount_titrate":                {"instr": DD_INSTR,             "render": discount_trial,       "extra_stages": ()},
    "psychological_refractory_period_two_choices": {"instr": PRP_INSTR, "render": prp_trial,           "extra_stages": ()},
    "ravens":                          {"instr": RAVENS_INSTR,         "render": ravens_trial,         "extra_stages": ()},
    "shift_task":                      {"instr": SHIFT_INSTR,          "render": shift_trial,          "extra_stages": ()},
    "digit_span":                      {"instr": DIGIT_INSTR,          "render": digit_trial,          "extra_stages": ()},
    "spatial_span":                    {"instr": SS_INSTR,             "render": spatial_span_trial,   "extra_stages": ()},
    "two_stage_decision":              {"instr": TWOSTAGE_INSTR,       "render": twostage_trial,       "extra_stages": ()},
    "columbia_card_task_cold":         {"instr": CCT_COLD_INSTR,       "render": cct_cold_trial,       "extra_stages": ()},
    "columbia_card_task_hot":          {"instr": CCT_HOT_INSTR,        "render": cct_hot_trial,        "extra_stages": ()},
    "angling_risk_task_always_sunny":  {"instr": ART_INSTR,            "render": art_trial,            "extra_stages": ()},
    "information_sampling_task":        {"instr": IST_INSTR,            "render": ist_trial,            "extra_stages": ("Decreasing_Win", "Fixed_Win")},
    "dietary_decision":                {"instr": DIET_INSTR,           "render": diet_trial,           "extra_stages": ("decision",)},
    "tower_of_london":                 {"instr": TOL_INSTR,            "render": tol_trial,            "extra_stages": ()},
    "keep_track":                      {"instr": KT_INSTR,             "render": kt_trial,             "extra_stages": ()},
}


REVERSE_NOTE = {
    "digit_span": DIGIT_INSTR_REVERSE_NOTE,
    "spatial_span": SS_INSTR_REVERSE_NOTE,
}


# ===========================================================================
# Driver
# ===========================================================================
def load_task(task: str, source: str) -> pd.DataFrame:
    fp = DATA_ROOTS[source] / f"{task}.csv.gz"
    df = pd.read_csv(fp, compression="gzip", low_memory=False)
    spec = TASKS[task]
    stages = ("test",) + spec.get("extra_stages", ())
    df = df[df["exp_stage"].isin(stages)].copy()
    df = df.dropna(subset=["worker_id"]).reset_index(drop=True)
    # Drop instruction-row leftovers for discount_titrate
    if task == "discount_titrate":
        df = df[df["choice"].notna()].reset_index(drop=True)
    if task == "psychological_refractory_period_two_choices":
        df = df[df["choice1_stim"].notna() & df["choice2_stim"].notna()].reset_index(drop=True)
    if task == "information_sampling_task":
        df = df[df["trial_id"].isin(["stim", "choice"])].reset_index(drop=True)
    if task == "angling_risk_task_always_sunny":
        df = df[df["trial_id"].isin(["stim", "round_over"])].reset_index(drop=True)
    if task == "dietary_decision":
        df = df[df["coded_response"].notna()].reset_index(drop=True)
    if task == "columbia_card_task_hot":
        # Collapse the per-click rows to one summary row per round: number of
        # safe gain cards turned, whether the round busted on a loss card, and
        # the round's risk parameters / outcome.
        df = df.sort_values("time_elapsed")
        is_card = pd.to_numeric(df["mouse_click"], errors="coerce").notna()
        df["_safe_card"] = is_card & (df["clicked_on_loss_card"] != 1)
        df = df.groupby(["worker_id", "which_round"], as_index=False, sort=False).agg(
            time_elapsed=("time_elapsed", "min"),
            gain_amount=("gain_amount", "last"),
            loss_amount=("loss_amount", "last"),
            num_loss_cards=("num_loss_cards", "last"),
            round_points=("round_points", "last"),
            n_cards_turned=("_safe_card", "sum"),
            busted=("clicked_on_loss_card", lambda s: int((s == 1).any())),
        )
    return df


def render_session(task: str, sub_df: pd.DataFrame) -> str:
    spec = TASKS[task]
    sub_df = sub_df.sort_values("time_elapsed").reset_index(drop=True)
    header = spec["instr"]
    if task in REVERSE_NOTE:
        if "reverse" in sub_df.get("condition", pd.Series()).astype(str).values:
            header = header + REVERSE_NOTE[task]
    lines = [header, ""]
    for _, row in sub_df.iterrows():
        try:
            lines.append(spec["render"](row))
        except Exception as e:
            lines.append(f"(skipped trial: {e})")
    return "\n".join(lines)


def write_task(task: str, source: str, full: bool) -> dict:
    df = load_task(task, source)
    out_subdir = OUT_DIR / source
    out_subdir.mkdir(exist_ok=True)
    suffix = "all" if full else "demo"
    out_fp = out_subdir / f"{task}.{suffix}.jsonl"

    subjects = sorted(df["worker_id"].astype(str).unique())
    if not full:
        subjects = subjects[:1]

    n_lines = 0
    total_chars = 0
    with out_fp.open("w", encoding="utf-8") as f:
        for s in subjects:
            sub_df = df[df["worker_id"].astype(str) == s]
            text = render_session(task, sub_df)
            f.write(json.dumps({"worker_id": s, "task": task, "source": source, "text": text}) + "\n")
            n_lines += 1
            total_chars += len(text)

    # Demo preview .txt for first subject
    if not full and subjects:
        txt_fp = out_subdir / f"{task}.demo.txt"
        with txt_fp.open("w", encoding="utf-8") as f:
            sub_df = df[df["worker_id"].astype(str) == subjects[0]]
            text = render_session(task, sub_df)
            f.write(f"===== {task} / {source} / subject {subjects[0]} =====\n")
            f.write(text)
            f.write("\n")

    return {
        "task": task,
        "source": source,
        "subjects": n_lines,
        "avg_chars": (total_chars // n_lines) if n_lines else 0,
        "out": str(out_fp),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=list(DATA_ROOTS), default="complete")
    p.add_argument("--full", action="store_true", help="Write all subjects (default: 1 demo subject)")
    p.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks to render")
    args = p.parse_args()

    tasks = args.tasks or list(TASKS)
    for task in tasks:
        try:
            s = write_task(task, args.source, args.full)
            print(f"  {s['task']:<48} {s['source']:<10} {s['subjects']:>5} subj  avg {s['avg_chars']:>6} chars  -> {s['out']}")
        except Exception as e:
            print(f"  {task}: ERROR {e}")


if __name__ == "__main__":
    main()
