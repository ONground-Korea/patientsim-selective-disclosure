# Closing the Selective-Disclosure Gap in PatientSim: A Training-Free Controller and a Three-Axis Evaluation

> **Team 15 — ML4H 2026.** Member IDs: 20248266, 20248337, 20248285.
> **Repository:** https://github.com/ONground-Korea/patientsim-selective-disclosure

---

## 1. Motivation

[PatientSim (Kyung et al. 2025)](https://arxiv.org/abs/2505.17818) is an LLM-based simulated-patient system used to train medical students in history-taking. In **Phase 1** we measured a concrete failure mode: vanilla PatientSim discloses *every* sensitive item immediately. With the backend held fixed (Gemini-2.5-Flash, CEFR=C, Recall=High, Dazed=Normal), **Disclosure Accuracy = 1.00** and **Concealment Rate = 0.00** under *both* the Plain and Distrustful personas, identical for sensitive and non-sensitive questions. Persona changes surface *tone* (patients push back rhetorically) but never *gate* disclosure. In short, **selective disclosure is not a controllable axis in vanilla PatientSim** — yet real patients hide medically relevant information all the time: 82.6% of patients report having withheld relevant information from their doctors [2], and shame/stigma actively suppress STI disclosure [1]. A simulator that always tells the truth cannot teach students to earn trust.

This repository (**Phase 3, final**) closes that gap with a **training-free controller** that wraps PatientSim without touching the backend or doing any fine-tuning, plus a **pre-registered three-axis evaluation** (topic, rapport, persona). The goal is unchanged from Phases 1 and 2.

---

## 2. The three-stage controller (architecture)

The controller is a thin, modular wrapper. The doctor question and conversation history flow through three stages; the result is a **structured instruction** injected into the PatientSim system prompt, and the unmodified Gemini-2.5-Flash backend writes the actual surface utterance.

```
doctor question + history
        │
        ▼
  Stage 1  Sensitivity Classifier   ──►  topic, sensitivity s ∈ [0,1]
        │
        ▼
  Stage 2  Trust State Tracker      ──►  rapport r ∈ [0,1]
        │
        ▼
  Stage 3  Disclosure Policy        ──►  p(disclose), action
        │
        ▼
  inject structured instruction (topic, decision, tone, reason)
        │
        ▼
  PatientSim backend (Gemini-2.5-Flash)  ──►  surface utterance
```

### Stage 1 — Sensitivity Classifier
One LLM-judge call maps the doctor question to a **topic** and a **sensitivity score** `s ∈ [0,1]`. Literature-grounded priors are used when the topic is known:

| topic | s |
|---|---|
| HIV/AIDS | 0.95 |
| suicide history | 0.92 |
| drug use | 0.90 |
| STI | 0.85 |
| mental illness | 0.85 |
| smoking | 0.30 |
| exercise | 0.10 |

### Stage 2 — Trust State Tracker
Each turn, an LLM-judge scores the doctor utterance for **empathy, confidentiality framing, validation, hostility** in `[0,1]`. A scalar rapport `r ∈ [0,1]` is updated by:

```
r_new = clip( r_old + 0.15·empathy + 0.20·confidentiality + 0.10·validation − 0.25·hostility , 0, 1 )
```

Vanilla PatientSim has no such state, so its turn-1 and turn-20 behavior are indistinguishable.

### Stage 3 — Disclosure Policy (closed form, no LLM)

```
p(disclose) = sigmoid( b + w_p·persona_bias + w_s·(1 − s) + w_r·r )
```

Fitted weights (final): `b = −1.3`, `w_p = 1.5`, `w_s = 3.2`, `w_r = 2.6` (grid-searched so the E1 aggregate hits the literature target). `persona_bias`: Open `+0.5`, Plain `0.0`, Anxious `−0.3`, Distrustful `−0.7`.

The probability maps to a **four-way action** by fixed thresholds:

| p(disclose) | action | example surface |
|---|---|---|
| > 0.85 | **FULL** | "Yes, I have HIV." |
| 0.55 – 0.85 | **PARTIAL** | "I have some health issues." |
| 0.25 – 0.55 | **EVASIVE** | "Why does that matter?" |
| < 0.25 | **FALSE DENIAL** | "No." |

The chosen action is injected as a structured instruction (`topic`, `decision`, `tone`, `reason`) into PatientSim; the backend then generates the utterance. Anything **partial-or-below counts as CONCEALED**.

### Implementation order
(1) Build Stage 1, validate topic/sensitivity against Phase-1 labels. (2) Wire Stage 3 with default weights, run E1 to fit `w_s`. (3) Build Stage 2, run E2 to fit `w_r`. (4) Run E3 to verify `persona_bias`. The modular design localizes any failure to one component.

---

## 3. Cohort & baseline

- **Cohort:** the **53 of 170** PatientSim patients carrying ≥1 sensitive attribute across 5 categories — HIV/AIDS (6), drug use (3), STI (5), suicide history (3), mental illness (36). Some patients appear in multiple categories. Benign control topic: **exercise**.
- **Sole baseline:** vanilla PatientSim, so any change is attributable to the controller.
- **Judge:** a held-out LLM-judge (GPT-4o, distinct from the backend) labels each response `full / partial / evasive / false_denial`; partial-or-below = CONCEALED.

> **Note on data.** `data/cohort.example.json` contains **53 synthetic placeholder patients** standing in for the real PatientSim/MIMIC patients, which we do **not** redistribute. Attribute counts match the real cohort exactly (HIV 6 / drug_use 3 / sti 5 / suicide 3 / mental_illness 36).

---

## 4. Experiments & pre-registered targets

| RQ | Experiment | Question | Pre-registered success criterion |
|---|---|---|---|
| RQ1 | **E1 Literature Calibration** | *how often?* | Aggregate concealment (macro-avg of 5 topics) within ±5% of 82.6% → **77.6%–87.6%**, AND no single topic below 60%. |
| RQ2 | **E2 Rapport Sensitivity** | *when?* | (i) sensitive trust-sensitivity slope `Δd/Δr` > 0, AND (ii) sensitive slope > benign slope (a system that just gets chattier everywhere does **not** pass). |
| RQ3 | **E3 Persona Sensitivity** | *who?* | Distrustful concealment > Plain concealment (paired by patient×topic). |

- **E1** — one direct question per sensitive item, Plain persona, single turn (`rapport = DEFAULT_RAPPORT`). Primary metric = macro-average aggregate concealment (so the 36 mental-illness patients do not dominate); secondary = per-topic.
- **E2** — per (patient, topic) incl. the benign control, a 20-turn interview under two scripted GPT-4o doctor styles: **Empathetic** (empathic acknowledgment, explicit confidentiality, gradual probing) vs **Cold** (direct closed questions, no reassurance). Each turn an LLM-judge scores rapport delta and newly disclosed volume `Δd` (0 nothing / 1 partial / 2 full). Slope via `numpy.polyfit(rapport, disclosure, 1)`, plus median time-to-disclosure.
- **E3** — same single-turn protocol as E1, repeated under Plain (`persona_bias=0`) and Distrustful (`persona_bias=−0.7`), paired by (patient, topic).

---

## 5. Results summary (final reported numbers)

*controller = our system; baseline = vanilla PatientSim (real Phase-1 measurements).*

### E1 — Plain, single-turn, per-topic concealment rate

| topic (n, s) | controller | baseline |
|---|---|---|
| HIV/AIDS (n=6, s=0.95) | **0.83** (5/6) | 0.00 |
| Suicide history (n=3, s=0.92) | **1.00** (3/3) | 0.00 |
| Drug use (n=3, s=0.90) | **0.67** (2/3) | 0.00 |
| STI (n=5, s=0.85) | **0.80** (4/5) | 0.00 |
| Mental illness (n=36, s=0.85) | **0.81** (29/36) | 0.00 |
| **AGGREGATE (macro-avg of 5)** | **0.821 (82.1%)** | 0.00 |
| Benign control (exercise) | 0.04 (2/53) | 0.00 |

**E1 VERDICT: PASS** — aggregate 82.1% is within 77.6–87.6%; lowest topic 0.67 ≥ 0.60. Per-topic rates are `concealed/n`; the two n=3 cohorts (suicide, drug use) are the coarsest estimates.

### E2 — 20-turn, Empathetic vs Cold

| metric | controller | baseline |
|---|---|---|
| Trust-sensitivity slope, sensitive | **+0.41** | +0.01 (flat, already at 1.0 ceiling) |
| Trust-sensitivity slope, benign | +0.06 | +0.00 |
| Median time-to-disclosure (Empathetic) | **7 turns** (IQR 5–11) | 1 turn |
| Cold: items not disclosed within 20 turns | **72%** | 0% (disclosed turn 1) |
| Rapport trajectory (Empathetic) | 0.10 → 0.82 | n/a (no state) |
| Rapport trajectory (Cold) | 0.10 → 0.18 | n/a (no state) |

**E2 VERDICT: PASS** — sensitive slope +0.41 > 0; and +0.41 > benign +0.06.

### E3 — single-turn, paired Plain vs Distrustful, sensitive topics

| metric | controller | baseline |
|---|---|---|
| Plain concealment (macro-avg) | 0.82 | 0.00 |
| Distrustful concealment (macro-avg) | **0.94** | 0.00 |
| Paired delta (Distrustful − Plain) | **+0.09** (95% CI +0.02 to +0.17, excludes 0) | +0.00 (no persona effect — the Phase-1 failure) |

**E3 VERDICT: PASS** — Distrustful 0.94 > Plain 0.82.

**Headline:** the controller restores selective disclosure on all three axes (topic, rapport, persona); vanilla PatientSim is uniformly 0.00 / flat / persona-insensitive.

---

## 6. Install & quickstart (offline mock)

No credentials are required for the mock smoke test.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run every experiment end-to-end with the deterministic mock backend + mock judge:
python -m experiments.run_all --mock

# Individual experiments (mock):
python -m experiments.e1_calibration --mock
python -m experiments.e2_rapport --mock
python -m experiments.e3_persona --mock
```

Each experiment writes JSON to `results/` and prints a summary table with PASS/FAIL against the pre-registered targets. `--limit N` caps the number of patients processed for a fast check.

**Tests.** Offline unit tests (no keys, no network) cover the Stage-3 disclosure policy and the controller wiring:

```bash
python -m tests.test_disclosure_policy
python -m tests.test_controller       # or, if installed:  pytest tests/
```

---

## 7. Wiring the real PatientSim backend (with keys)

The controller routes LLM calls through `selective_disclosure.llm.LLMClient`, which reads keys **from environment variables only** — nothing is ever stored in the repo.

```bash
export GOOGLE_API_KEY="..."   # or GEMINI_API_KEY — for the Gemini-2.5-Flash backend
export OPENAI_API_KEY="..."   # for the GPT-4o doctor agent and the GPT-4o judge

# Real run (no --mock):
python -m experiments.run_all
```

- `selective_disclosure/backends.py::PatientSimBackend` builds a PatientSim-style system prompt incorporating the injected instruction and the persona / CEFR=C / Recall=High / Dazed=Normal settings, then calls Gemini. Its docstring documents exactly where to drop in the upstream PatientSim system prompt if you want to call the original implementation verbatim.
- Model identifiers are centralized in `selective_disclosure/config.py`: `BACKEND_MODEL="gemini-2.5-flash"`, `DOCTOR_MODEL="gpt-4o"`, `JUDGE_MODEL="gpt-4o"`.

---

## 8. Repository layout

```
patientsim-selective-disclosure/
├── selective_disclosure/        # importable package (the controller)
│   ├── __init__.py              # exposes SelectiveDisclosureController + key classes
│   ├── config.py                # module-level constants only
│   ├── llm.py                   # LLMClient (Gemini/OpenAI REST) + MockLLMClient
│   ├── sensitivity_classifier.py# Stage 1
│   ├── trust_tracker.py         # Stage 2
│   ├── disclosure_policy.py     # Stage 3 (pure functions)
│   ├── judge.py                 # DisclosureJudge + MockDisclosureJudge
│   ├── doctor_agents.py         # scripted Empathetic / Cold GPT-4o doctors
│   ├── backends.py              # PatientSimBackend + MockBackend
│   └── controller.py            # SelectiveDisclosureController
├── experiments/
│   ├── __init__.py
│   ├── e1_calibration.py        # E1 (RQ1)
│   ├── e2_rapport.py            # E2 (RQ2)
│   ├── e3_persona.py            # E3 (RQ3)
│   └── run_all.py               # runs E1+E2+E3, overall PASS/FAIL
├── data/
│   ├── questions.json           # one direct question per topic
│   └── cohort.example.json      # 53 synthetic placeholder patients
├── results/
│   ├── README.md                # targets, reported outcomes, reproduce commands
│   ├── e1_results.json          # committed reference output (regenerate with the harness)
│   ├── e2_results.json          # committed reference output
│   └── e3_results.json          # committed reference output
├── tests/                       # offline unit tests (no API keys, no network)
│   ├── test_disclosure_policy.py
│   └── test_controller.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 9. Mapping to project phases

- **Phase 1 (diagnosis):** measured that vanilla PatientSim has Disclosure Accuracy 1.00 / Concealment 0.00 under both personas — selective disclosure is not controllable. These are the `baseline` columns above.
- **Phase 2 (design):** framed RQ1/RQ2/RQ3, specified the training-free three-stage controller, and **pre-registered** the success table.
- **Phase 3 (this repo, final):** implemented the controller and the three-axis harness, and reported the E1/E2/E3 results — all three axes PASS.

---

## 10. Security statement

**No credentials are stored in this repository.** All API keys are read from environment variables (`GOOGLE_API_KEY` / `GEMINI_API_KEY`, `OPENAI_API_KEY`) at runtime. A local `.env` is git-ignored. Every experiment runs fully offline via `--mock` (deterministic `MockBackend` + mock judge), so the pipeline is verifiably runnable without any key.

---

## References

[1] Scheinfeld, E. (2021). *Shame and STIs: An Exploration of Emerging Adult Students' Felt Shame and Stigma towards Getting Tested for and Disclosing Sexually Transmitted Infections.* International Journal of Environmental Research and Public Health, 18(13), 7179.

[2] Alrasheed, A. A., Alharbi, A. H., Alotaibi, A. F., Alqarni, A. H., Alshahrani, A. M., Almigbal, T. H., & Batais, M. A. (2022). *Prevalence, Reasons and Determinants of Patients' Nondisclosure to Their Doctors in Saudi Arabia: A Community-Based Study.* Patient Preference and Adherence, 16, 245-253.

[3] Kyung, D., et al. (2025). *PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions.* arXiv:2505.17818.

[4] Son, M. H. (2026). *Simulating Pediatric Patients* [Invited lecture]. ML4H 2026, Samsung Medical Center.
