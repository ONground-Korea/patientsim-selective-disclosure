"""Project-wide configuration constants for the selective-disclosure controller.

This module contains *module-level constants only*. It deliberately holds no
logic, no I/O, and no API keys. Every other module in the
:mod:`selective_disclosure` package imports its tunable parameters from here so
that there is a single, auditable source of truth for the numbers reported in
the ML4H 2026 Phase-3 paper
("Closing the Selective-Disclosure Gap in PatientSim: A Training-Free
Controller and a Three-Axis Evaluation").

The values below are the *final* fitted weights and literature-grounded priors
described in the report. Changing them changes the behaviour of the whole
pipeline; the experiment harnesses read these constants directly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage 1 - Sensitivity classifier priors
# ---------------------------------------------------------------------------
# Literature-grounded sensitivity priors s in [0, 1] for each topic key. These
# are used by the SensitivityClassifier whenever the LLM picks a known topic;
# unknown topics fall back to an LLM-estimated float in [0, 1].
SENSITIVITY_PRIORS: dict[str, float] = {
    "hiv": 0.95,
    "suicide": 0.92,
    "drug_use": 0.90,
    "sti": 0.85,
    "mental_illness": 0.85,
    "smoking": 0.30,
    "exercise": 0.10,
}

# ---------------------------------------------------------------------------
# Stage 3 - Disclosure policy: persona bias term
# ---------------------------------------------------------------------------
# persona_bias entries enter the disclosure-probability sigmoid additively
# (scaled by w_persona). A more open persona discloses more; a distrustful one
# conceals more.
PERSONA_BIAS: dict[str, float] = {
    "open": 0.5,
    "plain": 0.0,
    "anxious": -0.3,
    "distrustful": -0.7,
}

# ---------------------------------------------------------------------------
# Stage 2 - Trust state tracker: rapport update weights
# ---------------------------------------------------------------------------
# Each turn the trust tracker scores the doctor utterance for four cues in
# [0, 1] and updates a scalar rapport r via:
#     r_new = clip(r_old
#                  + empathy        * RAPPORT_WEIGHTS["empathy"]
#                  + confidentiality* RAPPORT_WEIGHTS["confidentiality"]
#                  + validation     * RAPPORT_WEIGHTS["validation"]
#                  + hostility      * RAPPORT_WEIGHTS["hostility"], 0, 1)
RAPPORT_WEIGHTS: dict[str, float] = {
    "empathy": 0.15,
    "confidentiality": 0.20,
    "validation": 0.10,
    "hostility": -0.25,
}

# ---------------------------------------------------------------------------
# Stage 3 - Disclosure policy: sigmoid weights
# ---------------------------------------------------------------------------
# Closed-form disclosure probability:
#     p = sigmoid(intercept
#                 + w_persona * persona_bias
#                 + w_s       * (1 - sensitivity)
#                 + w_r       * rapport)
# These weights were grid-searched so the E1 aggregate concealment rate hits the
# literature target of 82.6%.
POLICY_WEIGHTS: dict[str, float] = {
    "intercept": -1.3,
    "w_persona": 1.5,
    "w_s": 3.2,
    "w_r": 2.6,
}

# ---------------------------------------------------------------------------
# Stage 3 - Disclosure policy: four-way action thresholds on p
# ---------------------------------------------------------------------------
# Interpretation (applied in disclosure_policy.decide_action):
#     p >  full      -> FULL          ("Yes, I have HIV.")
#     p >= partial   -> PARTIAL       ("I have some health issues.")
#     p >= evasive   -> EVASIVE       ("Why does that matter?")
#     otherwise      -> FALSE_DENIAL  ("No.")
DISCLOSURE_THRESHOLDS: dict[str, float] = {
    "full": 0.85,
    "partial": 0.55,
    "evasive": 0.25,
}

# ---------------------------------------------------------------------------
# Topic groupings
# ---------------------------------------------------------------------------
# The five sensitive categories evaluated in the paper and the benign controls
# (exercise is the primary benign control; smoking is a low-sensitivity probe).
SENSITIVE_TOPICS: list[str] = ["hiv", "suicide", "drug_use", "sti", "mental_illness"]
BENIGN_TOPICS: list[str] = ["exercise", "smoking"]

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
# The PatientSim backend is held fixed at Gemini-2.5-Flash. The scripted doctor
# agent and the held-out disclosure judge both use GPT-4o (distinct from the
# backend, so judging is independent of generation).
BACKEND_MODEL: str = "gemini-2.5-flash"
DOCTOR_MODEL: str = "gpt-4o"
JUDGE_MODEL: str = "gpt-4o"

# Initial rapport every interview starts from (cold open).
DEFAULT_RAPPORT: float = 0.1

# ---------------------------------------------------------------------------
# Human-readable topic labels (for printing tables / writing results)
# ---------------------------------------------------------------------------
TOPIC_LABELS: dict[str, str] = {
    "hiv": "HIV/AIDS",
    "suicide": "Suicide history",
    "drug_use": "Drug use",
    "sti": "STI",
    "mental_illness": "Mental illness",
    "smoking": "Smoking",
    "exercise": "Exercise",
}
