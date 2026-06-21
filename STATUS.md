# SRO × Minitaur — project status

_Cross-task individual transfer: can a person's trial-level behavior on task A
predict their behavior on task B, captured by a behavioral foundation model?_

Repo: https://github.com/YifeiCAO/sro-minitaur-transfer · Base model:
`marcelbinz/Llama-3.1-Minitaur-8B` (Llama-3.1-8B + Psych-101, choice-only).

---

## 1. Setup

- **Data:** Self-Regulation Ontology (Eisenberg 2019). 522 subjects (train 418 /
  heldout 104) + 151 retest (time2). Currently everything uses **time1 only**.
- **Format:** Centaur-style NL transcripts; loss/prediction only on the human's
  `<<response>>` choice tokens. **Choice-only — RT is not modeled.**
- **Starting subset (11 tasks, 4 domains):** discounting (kirby, bickel,
  discount_titrate), risk (CCT cold/hot, information_sampling), RL
  (two_stage_decision, probabilistic_selection), conflict/memory-control
  (directed_forgetting, recent_probes, dot_pattern_expectancy).

## 2. Core claim & metric

Individual transfer rides on the **shuffled control**, not raw NLL: a person's
own source embedding must beat a *random other person's*. Primary metric =
cross-task **identification** (rank-based, dodges NLL dilution); chance = 1/K.

## 3. Results so far

| Phase | What | Result |
|------|------|--------|
| **0c handcrafted matrix** | transfer via scalar DVs (DDM/accuracy, incl. RT), reliability-normalized | **within-domain 0.353, across 0.054** (n=20/90). Signal exists; strongest: kirby→discount 0.85, CCT cold↔hot ~0.55, DF↔recent_probes ~0.5 |
| **1 — M_pop** | population fine-tune (no individual), per-response acc on heldout | macro **0.75**; predicts most tasks ≫ majority baseline (discount .89, DF .93, recent_probes .91, DPX .95; weak: two_stage .63, columbia .26–.36) |
| **2 — soft-prompt z-injection** | person-encoder + soft prompt into frozen M_pop | **FAILED** — NaN / collapse-to-chance / flat. Wrong rep + fragile mechanism (see §4) |
| **surprise-rep diagnostic** | DF→recent_probes, training-free | **top1 0.18 vs 0.10 chance (~2.8 SD, p≈0.002)** — real, modest |
| **surprise-rep matrix (11 tasks)** | per-person surprise profiles, all pairs | **within 0.153 > across 0.115 > chance 0.10** — right structure, modest, cell-level noisy |
| **in-context zero-shot** | put A's transcript before B, no training | **null** — real ≈ floor ≈ shuffled. Minitaur ignores cross-task context zero-shot |
| **in-context transfer matrix (within-domain)** | [A+B] fine-tune; identification top-1 (chance 0.10) on heldout, all within-domain pairs | ✅ **significant person-specific transfer across genuinely different tasks:** discounting 6/6 pairs id **0.20–0.37** (kirby/bickel/discount = 3 different paradigms); risk **CCT cold↔hot 0.32/0.40**; conflict **DF↔recent_probes 0.16/0.17** (p~1e-6). **Only clear within-domain null: CCT↔IST** (≈chance). **Degenerate/unmeasurable (NOT nulls):** RL two_stage_decision (too long → A truncated away) and ALL dot_pattern_expectancy cells (returns nan → DPX `<<>>` response tokens not matched = encoder bug to fix). No ID leakage; 0% split contamination. Headline stat = permutation null (t-test was variance-inflated). _Claim scoped to "within-domain own-session NLL advantage" pending base-rate control (see §3.5)_ |
| **retest ceiling** | same-task time1→time2 identification (upper bound) | **running / pending** — needed to interpret the magnitude |

### 3.5 Adversarial audit (6-agent, code+data re-verified)

**No bug. Result real as a mechanism.** Independently confirmed: NLL/masking/causal-shift
correct & symmetric (floor/real/shuffled differ only by prepended A); split clean (0
heldout in train); shuffled control fair; no stimulus/ID shortcut; p self-consistent (dz≈1.1).

**Two things to fix before saying "transfer":**
1. **Biggest risk — base-rate vs process.** A,B share response vocab → own-A could just
   leak the person's marginal choice rate. **DF→recent_probes largely escapes** (balanced
   target, 0% degenerate) → lead with it. **kirby→discount exposed** (23% near-degenerate)
   → run the **marginal-matched control** (`run_baserate_control.py`): real<matched ⇒ beyond base rate.
2. **Statistic — t-test inflated** (variance-deflated, non-independent shuffled arm). Switch to
   **sign-flip / exchangeability permutation** + dz + bootstrap CI + Wilcoxon (`stats.py`,
   `analyze_incontext_stats.py`, all offline). Report n_shuffle=1, ≥3 seeds, BH-FDR over pairs.

Strongest defensible phrasing today: _"within a cognitive domain, conditioning a choice-only
FM on a person's own prior session lowers their next-task choice NLL more than a stranger's."_
Items #1 (base-rate) + within>across upgrade this to "transfer."

## 4. Key insight (why soft-prompt failed, why surprise works)

Because Minitaur computes loss **only on choice tokens**, M_pop is a model of the
**population's** choice distribution. Its hidden states encode the *population
expectation given the stimulus* — shared across people, **person-invariant**. The
individual lives only in the **residual**: how surprised M_pop is by the person's
actual choices (NLL of their responses = "surprise profile").

- Soft-prompt used **mean hidden states** as the person-rep → person-invariant →
  nothing to inject → flat / unstable.
- Switching the rep to the **surprise profile** (the residual) → signal appears
  (0.18 diagnostic, 0.15 matrix).
- Corollary: a choice-only FM's individual signal is **inherently thin** (only the
  choice residual; RT invisible) → ~0 on near-deterministic tasks (discounting
  choices saturated, floor NLL 0.05), larger where choices vary (DF/recent_probes).

## 5. Honest read

- **Structure replicated:** surprise reps show within > across > chance, like the
  handcrafted 0c — the LLM residual carries the transfer structure.
- **Modest & noisy:** within 0.15 (~1.5× chance); single split; some within-domain
  cells at chance (e.g. CCT_cold→IST 0.038). Not directly comparable to 0c's 0.35
  (different metric: identification vs reliability-normalized CV-r).
- **Coherent thesis:** handcrafted (RT-containing) transfers strongly; choice-only
  FM captures a weaker version → consistent with much of the transferable signal
  being RT-based and invisible to the model.

## 6. Next steps

1. **retest ceiling** (running) — contextualizes 0.15 (transfer / ceiling).
2. **in-context fine-tune eval** (running) — does teaching cross-task context beat
   the surprise-rep? real-A < shuffled-A is the test.
3. **multi-seed the surprise matrix** — error bars on within vs across (cheap).
4. **same-metric 0c vs surprise** — quantify the RT gap.
5. **time1→time2 transfer** — clean trait test (removes within-occasion confound),
   using retest as the heldout pool.
6. **full 33-task matrix** (running on spare GPU) — complete transfer structure.

## 7. Outcomes (both publishable)

- **Positive:** identification > chance, real < shuffled, structure recovers the
  SRO ontology → trial-level sequences carry transferable individual structure.
- **Null:** even in-distribution / high-headroom pairs ≈ chance → clean model-based
  evidence (with reliability ceiling) that choice-level individual differences
  don't transfer across tasks — extends Eisenberg 2019 and answers the
  persona-induction negative in Binz et al. 2026.

_Potential collaborators identified (not yet public) — adjacent to the Binz 2026 /
ICLR 2026 "RL to explain human decisions" line of work._
