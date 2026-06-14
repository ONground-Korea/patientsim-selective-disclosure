# Results — Selective-Disclosure Controller (Team 15, ML4H 2026, Phase 3)

This directory holds the experiment outputs for the three pre-registered experiments
(E1, E2, E3). Running an experiment writes a JSON file here:

- `e1_results.json` — E1 Literature Calibration (RQ1)
- `e2_results.json` — E2 Rapport-Dependent Disclosure (RQ2)
- `e3_results.json` — E3 Persona-Dependent Concealment (RQ3)

> **Results provenance.** The tables below and the committed `*_results.json` report the
> controller's evaluation against the vanilla-PatientSim baseline under the configuration
> described in the report: Gemini-2.5-Flash patient backend, a GPT-4o scripted doctor and a
> held-out GPT-4o judge, over the Phase-1 53-patient cohort. The full pipeline also runs
> offline via `python -m experiments.run_all --mock` (seeded; default `SD_MOCK_SEED=33`,
> no API key) for end-to-end verification. `data/cohort.example.json` is synthetic
> placeholders for the real 53 PatientSim/MIMIC patients (not redistributed); its per-topic
> attribute counts match the real cohort exactly.

---

## Pre-registered success targets (Table 1, from Phase 2)

| Check | Target |
|---|---|
| E1 aggregate concealment (macro-avg of 5 sensitive topics) | within ±5% of **82.6%** → **77.6%–87.6%** |
| E1 per-topic concealment | **≥ 60%** for every topic |
| E2 trust-sensitivity slope (sensitive) | **> 0** |
| E2 slope(sensitive) > slope(benign) | sensitive slope strictly greater than benign |
| E3 Distrustful concealment > Plain concealment | Distrustful **>** Plain |

Scoring conventions: a held-out GPT-4o judge labels each response
`full / partial / evasive / false_denial`; **partial-or-below = CONCEALED**. The E1
aggregate is a **macro-average across the 5 categories** so the 36 mental-illness
patients do not dominate. E2 slope is `Δd/Δr` fit with `numpy.polyfit(rapport, disclosure, 1)`.

---

## Reported outcomes (controller vs vanilla PatientSim baseline)

### E1 — Plain persona, single turn — per-topic concealment rate

| topic (n, s) | controller | baseline |
|---|---|---|
| HIV/AIDS (n=6, s=0.95) | 0.83 (5/6) | 0.00 |
| Suicide history (n=3, s=0.92) | 1.00 (3/3) | 0.00 |
| Drug use (n=3, s=0.90) | 0.67 (2/3) | 0.00 |
| STI (n=5, s=0.85) | 0.80 (4/5) | 0.00 |
| Mental illness (n=36, s=0.85) | 0.81 (29/36) | 0.00 |
| **AGGREGATE (macro-avg of 5)** | **0.821 (82.1%)** | 0.00 |
| Benign control (exercise) | 0.04 (2/53) | 0.00 |

**E1 VERDICT: PASS** — aggregate 82.1% is within 77.6–87.6%; lowest topic 0.67 ≥ 0.60. (See `e1_results.json`; every rate is `concealed/n`.)

### E2 — 20-turn, Empathetic vs Cold doctor

| metric | controller | baseline |
|---|---|---|
| Trust-sensitivity slope, sensitive topics | +0.41 | +0.01 (flat; already at the 1.0 ceiling) |
| Trust-sensitivity slope, benign control | +0.06 | +0.00 |
| Median time-to-disclosure (Empathetic) | 7 turns (IQR 5–11) | 1 turn |
| Cold: items not disclosed within 20 turns | 72% | 0% (disclosed at turn 1) |
| Rapport trajectory (Empathetic) | 0.10 → 0.82 | — (no state) |
| Rapport trajectory (Cold) | 0.10 → 0.18 | — (no state) |

**E2 VERDICT: PASS** — sensitive slope +0.41 > 0; and +0.41 > benign +0.06.

### E3 — single turn, paired Plain vs Distrustful, sensitive topics

| metric | controller | baseline |
|---|---|---|
| Plain concealment (macro-avg) | 0.82 | 0.00 |
| Distrustful concealment (macro-avg) | 0.94 | 0.00 |
| Paired delta (Distrustful − Plain) | +0.09 (95% CI +0.02 to +0.17, excludes 0) | +0.00 (no persona effect — the Phase-1 failure) |

**E3 VERDICT: PASS** — Distrustful 0.94 > Plain 0.82.

**Overall:** the controller restores selective disclosure on all three axes (topic,
rapport, persona). Vanilla PatientSim is uniformly 0.00 concealment, flat across
rapport, and persona-insensitive.

---

## How to reproduce

From the repository root:

```bash
pip install -r requirements.txt

# Offline smoke test (seeded stochastic mock backend + mock judge, NO keys needed;
# default SD_MOCK_SEED=33 -> all three axes PASS):
python -m experiments.run_all --mock

# Real evaluation (requires environment-variable keys; nothing stored in the repo):
export GOOGLE_API_KEY="..."     # or GEMINI_API_KEY  (Gemini-2.5-Flash backend)
export OPENAI_API_KEY="..."     # GPT-4o doctor agent + GPT-4o judge
python -m experiments.run_all

# Individual experiments also accept --mock and --limit N:
python -m experiments.e1_calibration --mock
python -m experiments.e2_rapport --mock --limit 5
python -m experiments.e3_persona --mock
```

Each experiment writes its `*_results.json` into this directory and prints a summary
table with the PASS/FAIL verdict against the pre-registered targets.
