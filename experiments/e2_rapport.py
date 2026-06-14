"""E2 -- Rapport-Dependent Disclosure ("when"): RQ2.

Pre-registered question
-----------------------
Does the patient disclose *more* as the clinician builds trust?

Protocol
--------
* For every ``(patient, topic)`` pair -- including the benign ``exercise``
  control -- we run a **20-turn** interview under two scripted GPT-4o doctor
  styles:

  - **Empathetic**: empathic acknowledgment, explicit confidentiality,
    gradual probing (rapport should climb).
  - **Cold**: direct closed questions, no reassurance (rapport stays low).

* Each turn, the :class:`TrustTracker` updates the scalar rapport ``r`` from
  the doctor's utterance, and the controller produces a patient response that
  the held-out judge labels. We convert the label into a *disclosure volume*:

  ===============  ======
  judge label      volume
  ===============  ======
  ``false_denial``      0
  ``evasive``           0
  ``partial``           1
  ``full``              2
  ===============  ======

Metrics
-------
* **Trust-sensitivity slope** = ``Delta_d / Delta_r``, estimated with
  ``numpy.polyfit(rapport, disclosure, 1)`` over the pooled per-turn points.
* **Median time-to-disclosure**: the first turn at which a sensitive item is
  disclosed (volume >= 1), summarised across pairs.

Success (pre-registered, Table 1)
---------------------------------
* sensitive slope > 0; **and**
* sensitive slope > benign slope (so a system that merely gets *chattier*
  everywhere does **not** pass).

Baseline (vanilla PatientSim) is flat: it already discloses on turn 1, so its
slope is ~0 at the disclosure ceiling.

Run
---
.. code-block:: bash

    python -m experiments.e2_rapport --mock           # offline smoke test
    python -m experiments.e2_rapport                  # real run (needs keys)
    python -m experiments.e2_rapport --mock --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from selective_disclosure.config import (
    BENIGN_TOPICS,
    DOCTOR_MODEL,
    JUDGE_MODEL,
    SENSITIVE_TOPICS,
)
from selective_disclosure.controller import SelectiveDisclosureController
from selective_disclosure.doctor_agents import DoctorAgent
from selective_disclosure.judge import DisclosureJudge, MockDisclosureJudge
from selective_disclosure.llm import LLMClient, MockLLMClient

# --------------------------------------------------------------------------- #
# Experiment constants.                                                       #
# --------------------------------------------------------------------------- #
N_TURNS: int = 20
DOCTOR_STYLES: Tuple[str, str] = ("empathetic", "cold")
BENIGN_CONTROL_TOPIC: str = "exercise"

# Map judge labels to a newly-disclosed-volume score (0 nothing / 1 partial /
# 2 full), per the pre-registered scheme.
LABEL_TO_VOLUME: Dict[str, int] = {
    "false_denial": 0,
    "evasive": 0,
    "partial": 1,
    "full": 2,
}

# --------------------------------------------------------------------------- #
# Filesystem helpers (repo root is one level above experiments/).             #
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
_COHORT_PATH = os.path.join(_DATA_DIR, "cohort.example.json")
_QUESTIONS_PATH = os.path.join(_DATA_DIR, "questions.json")
_RESULTS_PATH = os.path.join(_RESULTS_DIR, "e2_results.json")


def _load_json(path: str) -> Any:
    """Load JSON from *path* with a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            "Run from the repository root so data/ is discoverable."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_cohort() -> List[Dict[str, Any]]:
    """Return the patient records from ``data/cohort.example.json``."""
    cohort = _load_json(_COHORT_PATH)
    if not isinstance(cohort, list):
        raise ValueError("cohort.example.json must contain a JSON list.")
    return cohort


def load_questions() -> Dict[str, str]:
    """Return the topic -> direct-question map from ``data/questions.json``."""
    questions = _load_json(_QUESTIONS_PATH)
    if not isinstance(questions, dict):
        raise ValueError("questions.json must contain a JSON object.")
    return questions


# --------------------------------------------------------------------------- #
# Builders.                                                                   #
# --------------------------------------------------------------------------- #
def _make_judge(mock: bool) -> DisclosureJudge:
    """Build the held-out disclosure judge (GPT-4o, distinct from the backend)."""
    if mock:
        return MockDisclosureJudge(MockLLMClient(model=JUDGE_MODEL))
    return DisclosureJudge(LLMClient(model=JUDGE_MODEL))


def _make_doctor(style: str, mock: bool) -> DoctorAgent:
    """Build a scripted GPT-4o :class:`DoctorAgent` for *style*."""
    llm = MockLLMClient(model=DOCTOR_MODEL) if mock else LLMClient(model=DOCTOR_MODEL)
    return DoctorAgent(llm, style=style)


# --------------------------------------------------------------------------- #
# Single (patient, topic, style) session.                                     #
# --------------------------------------------------------------------------- #
def run_session(
    controller: SelectiveDisclosureController,
    doctor: DoctorAgent,
    judge: DisclosureJudge,
    target_topic: str,
    has_attribute: bool,
    n_turns: int = N_TURNS,
) -> Dict[str, Any]:
    """Run one 20-turn interview and return per-turn rapport/disclosure.

    Returns
    -------
    dict
        ``{"rapport": [...], "disclosure": [...], "time_to_disclosure": int|None,
        "turns": [...]}``. ``time_to_disclosure`` is the 1-indexed turn of the
        first disclosure (volume >= 1), or ``None`` if never disclosed.
    """
    controller.reset()
    history: List[Dict[str, str]] = []
    rapport_series: List[float] = []
    disclosure_series: List[int] = []
    turn_log: List[Dict[str, Any]] = []
    time_to_disclosure: Optional[int] = None

    for turn_idx in range(1, n_turns + 1):
        # 1. Doctor speaks (style-dependent), conditioned on the dialogue.
        doctor_utterance = doctor.next_utterance(history, target_topic)

        # 2. Controller responds; this also updates the trust tracker so the
        #    reported rapport reflects the doctor's latest utterance.
        turn = controller.respond(doctor_utterance, history)
        rapport = float(turn["rapport"])

        # 3. Judge the patient response -> disclosure volume.
        label = judge.label(
            question=doctor_utterance,
            patient_response=turn["response"],
            has_attribute=has_attribute,
        )
        volume = LABEL_TO_VOLUME.get(label["label"], 0)

        if time_to_disclosure is None and volume >= 1:
            time_to_disclosure = turn_idx

        rapport_series.append(rapport)
        disclosure_series.append(int(volume))
        turn_log.append(
            {
                "turn": turn_idx,
                "doctor": doctor_utterance,
                "rapport": rapport,
                "p": turn["p"],
                "action": turn["action"],
                "label": label["label"],
                "volume": int(volume),
                "response": turn["response"],
            }
        )

        # 4. Append both utterances to the running history.
        history = history + [
            {"role": "doctor", "content": doctor_utterance},
            {"role": "patient", "content": turn["response"]},
        ]

    return {
        "topic": target_topic,
        "rapport": rapport_series,
        "disclosure": disclosure_series,
        "time_to_disclosure": time_to_disclosure,
        "turns": turn_log,
    }


# --------------------------------------------------------------------------- #
# Slope estimation.                                                           #
# --------------------------------------------------------------------------- #
def estimate_slope(rapport: List[float], disclosure: List[int]) -> float:
    """Return the OLS slope of disclosure on rapport via ``numpy.polyfit``.

    Falls back to ``0.0`` when there is no variation in rapport (a degenerate
    fit, e.g. a perfectly flat Cold interview), which keeps the metric stable.
    """
    if len(rapport) < 2:
        return 0.0
    x = np.asarray(rapport, dtype=float)
    y = np.asarray(disclosure, dtype=float)
    if float(np.ptp(x)) == 0.0:
        # No rapport movement -> slope is undefined; report flat.
        return 0.0
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _median(values: List[float]) -> Optional[float]:
    """Median of *values*, or ``None`` if empty."""
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _iqr(values: List[float]) -> Optional[Tuple[float, float]]:
    """Return the (Q1, Q3) inter-quartile range, or ``None`` if empty."""
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return (q1, q3)


# --------------------------------------------------------------------------- #
# Core experiment.                                                            #
# --------------------------------------------------------------------------- #
def run_e2(mock: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    """Execute experiment E2 and return a results dictionary.

    Parameters
    ----------
    mock:
        Run fully offline (no API keys) when ``True``.
    limit:
        Optional cap on the number of patients processed.
    """
    cohort = load_cohort()
    questions = load_questions()
    if limit is not None:
        cohort = cohort[: max(0, int(limit))]

    judge = _make_judge(mock=mock)

    # Two doctor agents, reused across sessions (stateless across calls).
    doctors = {style: _make_doctor(style, mock=mock) for style in DOCTOR_STYLES}

    # Pooled per-turn points, separated into sensitive vs benign and by style.
    pooled: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        style: {
            "sensitive": {"rapport": [], "disclosure": []},
            "benign": {"rapport": [], "disclosure": []},
        }
        for style in DOCTOR_STYLES
    }
    # Time-to-disclosure samples for sensitive items, per style.
    ttd: Dict[str, List[int]] = {style: [] for style in DOCTOR_STYLES}
    ttd_not_disclosed: Dict[str, int] = {style: 0 for style in DOCTOR_STYLES}
    ttd_total: Dict[str, int] = {style: 0 for style in DOCTOR_STYLES}

    sessions: List[Dict[str, Any]] = []

    for patient in cohort:
        patient_id = patient.get("patient_id")
        attributes = set(patient.get("attributes", []))

        # Build the list of (topic, kind) pairs to run for this patient:
        #   every sensitive topic the patient carries + the benign control.
        pairs: List[Tuple[str, str]] = []
        for topic in SENSITIVE_TOPICS:
            if topic in attributes:
                pairs.append((topic, "sensitive"))
        if BENIGN_CONTROL_TOPIC in BENIGN_TOPICS:
            pairs.append((BENIGN_CONTROL_TOPIC, "benign"))

        for topic, kind in pairs:
            if topic not in questions:
                continue
            for style in DOCTOR_STYLES:
                # Each session gets a fresh Plain-persona controller so rapport
                # starts at DEFAULT_RAPPORT every time.
                controller = SelectiveDisclosureController.build(
                    persona="plain", mock=mock
                )
                session = run_session(
                    controller=controller,
                    doctor=doctors[style],
                    judge=judge,
                    target_topic=topic,
                    has_attribute=True,
                    n_turns=N_TURNS,
                )
                session["patient_id"] = patient_id
                session["style"] = style
                session["kind"] = kind
                sessions.append(session)

                bucket = pooled[style]["sensitive" if kind == "sensitive" else "benign"]
                bucket["rapport"].extend(session["rapport"])
                bucket["disclosure"].extend(session["disclosure"])

                if kind == "sensitive":
                    ttd_total[style] += 1
                    if session["time_to_disclosure"] is None:
                        ttd_not_disclosed[style] += 1
                    else:
                        ttd[style].append(session["time_to_disclosure"])

    # ----------------------------------------------------------------- #
    # Slopes per style and bucket.                                      #
    # ----------------------------------------------------------------- #
    slopes: Dict[str, Dict[str, float]] = {}
    for style in DOCTOR_STYLES:
        slopes[style] = {
            "sensitive": estimate_slope(
                pooled[style]["sensitive"]["rapport"],
                pooled[style]["sensitive"]["disclosure"],
            ),
            "benign": estimate_slope(
                pooled[style]["benign"]["rapport"],
                pooled[style]["benign"]["disclosure"],
            ),
        }

    # Headline slopes pool all per-turn points (both styles) so the metric
    # captures the full rapport range from Cold-low to Empathetic-high.
    all_sensitive_rapport: List[float] = []
    all_sensitive_disclosure: List[float] = []
    all_benign_rapport: List[float] = []
    all_benign_disclosure: List[float] = []
    for style in DOCTOR_STYLES:
        all_sensitive_rapport.extend(pooled[style]["sensitive"]["rapport"])
        all_sensitive_disclosure.extend(pooled[style]["sensitive"]["disclosure"])
        all_benign_rapport.extend(pooled[style]["benign"]["rapport"])
        all_benign_disclosure.extend(pooled[style]["benign"]["disclosure"])

    sensitive_slope = estimate_slope(all_sensitive_rapport, all_sensitive_disclosure)
    benign_slope = estimate_slope(all_benign_rapport, all_benign_disclosure)

    # ----------------------------------------------------------------- #
    # Time-to-disclosure summary (focus on the Empathetic style).        #
    # ----------------------------------------------------------------- #
    ttd_summary: Dict[str, Any] = {}
    for style in DOCTOR_STYLES:
        median_ttd = _median([float(v) for v in ttd[style]])
        iqr = _iqr([float(v) for v in ttd[style]])
        total = ttd_total[style]
        not_disclosed = ttd_not_disclosed[style]
        ttd_summary[style] = {
            "median_turns_to_first_disclosure": median_ttd,
            "iqr": list(iqr) if iqr is not None else None,
            "n_sensitive_sessions": total,
            "n_not_disclosed_within_20": not_disclosed,
            "frac_not_disclosed_within_20": (
                (not_disclosed / total) if total else 0.0
            ),
        }

    # ----------------------------------------------------------------- #
    # Verdict against the pre-registered targets.                       #
    # ----------------------------------------------------------------- #
    sensitive_positive = sensitive_slope > 0.0
    sensitive_above_benign = sensitive_slope > benign_slope
    passed = bool(sensitive_positive and sensitive_above_benign)

    results: Dict[str, Any] = {
        "experiment": "E2",
        "name": "Rapport-Dependent Disclosure (RQ2)",
        "mock": bool(mock),
        "n_turns": N_TURNS,
        "styles": list(DOCTOR_STYLES),
        "benign_control": BENIGN_CONTROL_TOPIC,
        "slopes_by_style": slopes,
        "slope_sensitive": sensitive_slope,
        "slope_benign": benign_slope,
        "time_to_disclosure": ttd_summary,
        "verdict": {
            "sensitive_slope_positive": sensitive_positive,
            "sensitive_slope_gt_benign": sensitive_above_benign,
            "passed": passed,
        },
        "baseline_note": (
            "Vanilla PatientSim discloses on turn 1 under BOTH styles "
            "(time-to-disclosure = 1), so its slope is ~0 at the ceiling."
        ),
        "n_sessions": len(sessions),
    }
    return results


# --------------------------------------------------------------------------- #
# Output helpers.                                                             #
# --------------------------------------------------------------------------- #
def write_results(results: Dict[str, Any], path: str = _RESULTS_PATH) -> str:
    """Write *results* to *path* as pretty JSON, creating dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def print_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary for E2."""
    print("=" * 68)
    print("E2 -- Rapport-Dependent Disclosure (RQ2): does the patient")
    print("      disclose more as the clinician builds trust?")
    print("=" * 68)
    print(f"  mode: {'MOCK (offline)' if results['mock'] else 'REAL (API keys)'}")
    print(f"  turns/session: {results['n_turns']}  |  sessions: {results['n_sessions']}")
    print("-" * 68)
    print(f"  Trust-sensitivity slope (sensitive): {results['slope_sensitive']:+.3f}")
    print(f"  Trust-sensitivity slope (benign):    {results['slope_benign']:+.3f}")
    print("-" * 68)
    print("  Per-style slopes (sensitive / benign):")
    for style in results["styles"]:
        s = results["slopes_by_style"][style]
        print(f"    {style:<12}{s['sensitive']:+.3f}  /  {s['benign']:+.3f}")
    print("-" * 68)
    print("  Median time-to-first-disclosure (sensitive items):")
    for style in results["styles"]:
        t = results["time_to_disclosure"][style]
        med = t["median_turns_to_first_disclosure"]
        med_str = f"{med:.0f}" if med is not None else "n/a"
        iqr = t["iqr"]
        iqr_str = f"IQR {iqr[0]:.0f}-{iqr[1]:.0f}" if iqr else "IQR n/a"
        frac = t["frac_not_disclosed_within_20"]
        print(
            f"    {style:<12}{med_str:>4} turns  ({iqr_str}; "
            f"{frac*100:.0f}% not disclosed within {results['n_turns']})"
        )
    print("-" * 68)
    v = results["verdict"]
    print(f"  sensitive slope > 0:        {v['sensitive_slope_positive']}")
    print(f"  sensitive slope > benign:   {v['sensitive_slope_gt_benign']}")
    print(f"  E2 VERDICT: {'PASS' if v['passed'] else 'FAIL'}")
    print("=" * 68)


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the E2 CLI."""
    parser = argparse.ArgumentParser(
        prog="experiments.e2_rapport",
        description="E2 Rapport-Dependent Disclosure for the selective-disclosure controller.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run fully offline with MockBackend + MockDisclosureJudge (no API keys).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most LIMIT patients (for quick smoke tests).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Entry point: run E2, write results, print the summary, return results."""
    args = build_arg_parser().parse_args(argv)
    results = run_e2(mock=args.mock, limit=args.limit)
    out_path = write_results(results)
    print_summary(results)
    print(f"\nWrote results to: {out_path}")
    return results


if __name__ == "__main__":
    main()
