# SRO × Minitaur — Cross-task individual transfer

Can a person's **trial-level behavior on task A** predict that **same person's
trial-level behavior on task B**? I.e. are task-based individual differences
**transferable across tasks**, and can a behavioral foundation model
(Minitaur/Centaur — Llama-3.1-8B + Psych-101) capture that transfer?

Data: the **Self-Regulation Ontology** (Eisenberg et al., 2019), 522 subjects ×
~30 tasks, with a 150-subject retest sample.

## The claim rides on a control, not on raw NLL

Replace a person's source-task embedding `z` with a **random other person's**
embedding (`shuffled-z`). If real `z` beats `shuffled-z`, the signal is
**person-specific** cross-task transfer — and the result is immune to whether
`M_pop` is perfectly calibrated (both add identical capacity).

- **Primary result:** cross-task **identification** above chance (rank-based,
  dodges NLL dilution).
- **Secondary:** **NLL improvement over a population floor** (real-z vs floor vs
  shuffled-z).

Both a positive and a null result are publishable (see *Outcomes* below).

## Important design decision: choice-only

Minitaur predicts only the human's **choice** tokens (`<<...>>`), **not reaction
time**. This repo is built **choice-only**. Consequence, measured empirically
(`results/reliability`): the classic conflict tasks (stroop / simon / ANT) are at
the **accuracy ceiling** — their individual differences live in RT, which the
model cannot see — so they are **excluded** from choice-only transfer. The signal
lives in tasks with real choice variance: **discounting, risk, RL, value-based
self-control**, and the no-ceiling control tasks (**directed_forgetting,
recent_probes, DPX**). See `configs/tasks.yaml`.

## Pipeline

| Phase | What | Code | GPU |
|------|------|------|-----|
| 0a | data → Centaur NL (`<<response>>`) | `data/centaur_render.py` | no |
| 0b | test-retest **reliability ceiling** | `diagnostics/reliability.py` | no |
| 0c | **handcrafted transfer matrix** `T[A,B]` (decision gate) | `diagnostics/transfer_matrix.py` | no |
| 1 | **M_pop**: population fine-tune, then freeze (= no-individual floor) | `model/mpop.py` | yes |
| 2 | **person-encoder** `E_A` + injection into frozen M_pop | `model/{person_encoder,inject,transfer_model}.py` | yes |
| 3 | cross-task **identification** | `eval/identification.py` | yes |
| 4 | full model + **NLL ablations** (real/floor/shuffled-z) | `eval/nll.py` | yes |

**Run Phase 0b/0c first.** They need no GPU and gate the expensive work: if
`T[A,B] ≈ 0` everywhere, task-based individual differences barely transfer —
shrink scope before building the 8B rig.

## Repo layout

```
configs/        tasks.yaml (taxonomy + subsets), default.yaml (paths, hparams)
src/sro_transfer/
  data/         centaur_render.py (encoders), datasets.py, splits.py
  diagnostics/  reliability.py (0b), handcrafted.py + transfer_matrix.py (0c)
  model/        masking.py (<<>>-masked loss), mpop.py (1),
                person_encoder.py + inject.py + transfer_model.py (2)
  eval/         scoring.py, identification.py (3), nll.py (4)
scripts/        thin CLIs for each phase
notebooks/      Colab entry points
```

## Data lives on Google Drive (not in the repo)

The NL data is ~150 MB (ART alone is 74 MB), so it is **not committed**. Lay it
out on Drive and point `configs/default.yaml:paths.data_root` at it:

```
<data_root>/
  output_nl/
    complete/   <task>.all.jsonl + <task>.correct.jsonl   (522 subj)
    retest/     <task>.all.jsonl + <task>.correct.jsonl   (151 subj)
  sro_dv/
    complete.csv   # = SRO Data/Complete_02-16-2019/meaningful_variables.csv
    retest.csv     # = SRO Data/Retest_02-16-2019/meaningful_variables.csv
```

Regenerate the NL from a SRO checkout with `scripts/render_nl.py` if needed
(most users won't — generate once, keep on Drive).

## Quickstart (Colab)

```python
# 1. mount + clone + install
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/YifeiCAO/sro-minitaur-transfer.git
%cd sro-minitaur-transfer
!pip install -e .                       # core (CPU)

# 2. Phase 0b/0c — diagnostics, no GPU, gives the go/no-go signal
!python scripts/run_reliability.py     --config configs/default.yaml
!python scripts/run_transfer_matrix.py --config configs/default.yaml --subset starting_subset

# 3. Phase 1 — fine-tune M_pop (switch Colab runtime to GPU first)
!pip install -r requirements-model.txt
!python scripts/finetune_mpop.py --subset starting_subset --out results/mpop

# 4. Phase 3 — identification sanity (floor should sit at chance)
!python scripts/run_identification.py --mpop results/mpop --target two_stage_decision
```

The base model is **Centaur 8B** (`marcelbinz/Llama-3.1-Centaur-8B`, the plan's
"Minitaur"), loaded merged by default. It is gated: log in to HF and accept the
Llama license first (the notebook has a login cell). To use a LoRA-adapter
checkpoint instead, set `model.base_is_adapter: true` and point `base_model` at
the adapter.

## Baseline ladder (for reviewers)

1. domain-specific cognitive model (per task) — Centaur-style baseline
2. **M_pop** / frozen FM-on-SRO (no individual) — **floor**
3. **+ shuffled-z** — capacity control
4. **handcrafted transfer** (`T[A,B]`) — interpretable signal
5. **+ real-z (single-source)** — `T_model`
6. **+ real-z (multi-source)** — **full**

## Outcomes

- **Positive:** identification > chance; real-z < shuffled-z; `T_model` shows
  interpretable hubs that recover/exceed the SRO ontology structure →
  *trial-level sequences carry transferable individual structure that scalar
  summaries miss.*
- **Null:** even the in-distribution / high-reliability subset is ≈ shuffled-z
  and identification ≈ chance → *a flexible behavioral FM plus a reliability
  ceiling shows task-based individual differences do not transfer across tasks*
  — a clean model-based extension of Eisenberg et al. (2019).

## Caveats baked into the data

- `ravens` — opaque item ids (copyrighted matrices, no visual content); pure
  fluid-`g` outcome signal only.
- `keep_track` — the word stream is absent from the release; responses are
  near-unpredictable, signal ≈ recall span only.
- `angling_risk_task_always_sunny` — very long sequences (~1191 responses /
  subject); will be truncated at `model.max_seq_len`.
- ceiling tasks (`stroop`, `simon`, `attention_network_task`, …) — excluded from
  choice-only transfer.

## Status

Phase 0a–0c and the eval plumbing are runnable now. Phase 1 is real fine-tune
code (needs a GPU to validate). Phase 2 (`person_encoder` / `inject` /
`transfer_model`) is a working scaffold with two clearly-marked integration
seams. Build 0b/0c → read `T[A,B]` → decide how far to push the model stack.
