"""E1 -- Literature Calibration ("how often"): RQ1.

Pre-registered question
-----------------------
Does the controller's *per-topic concealment frequency* match patient
non-disclosure base rates from the literature?

Protocol
--------
* Cohort: the 53-patient Phase-1 set (synthetic placeholders in
  ``data/cohort.example.json``), spanning the five sensitive categories.
* For every patient and every *sensitive* topic that the patient actually
  carries, we ask the single direct question from ``data/questions.json``
  under the **Plain** persona, in a **single turn** (rapport pinned at
  ``DEFAULT_RAPPORT``).
* A held-out LLM judge (GPT-4o, distinct from the Gemini backend) labels each
  response ``full`` / ``partial`` / ``evasive`` / ``false_denial``;
  ``partial`` or below counts as **CONCEALED**.
* Benign control: the ``exercise`` topic is asked of *every* patient (the
  controller should almost never conceal it).

Primary metric
--------------
Aggregate concealment rate = the **macro-average** over the five sensitive
topics (so the 36 mental-illness patients do not dominate).

Success (pre-registered, Table 1)
---------------------------------
* Aggregate within +/-5% of the 82.6% literature base rate
  (i.e. ``0.776 <= aggregate <= 0.876``); **and**
* No single sensitive topic below ``0.60``.

Baseline (vanilla PatientSim, measured in Phase 1) conceals nothing:
every per-topic rate is ``0.00``.

Run
---
.. code-block:: bash

    python -m experiments.e1_calibration --mock          # offline smoke test
    python -m experiments.e1_calibration                 # real run (needs keys)
    python -m experiments.e1_calibration --mock --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from selective_disclosure.config import (
    DEFAULT_RAPPORT,
    SENSITIVE_TOPICS,
    TOPIC_LABELS,
)
from selective_disclosure.controller import SelectiveDisclosureController
from selective_disclosure.config import JUDGE_MODEL
from selective_disclosure.judge import DisclosureJudge, MockDisclosureJudge
from selective_disclosure.llm import LLMClient, MockLLMClient

# --------------------------------------------------------------------------- #
# Pre-registered targets (Table 1).                                           #
# --------------------------------------------------------------------------- #
LITERATURE_BASE_RATE: float = 0.826
AGGREGATE_TOLERANCE: float = 0.05
AGGREGATE_LOW: float = LITERATURE_BASE_RATE - AGGREGATE_TOLERANCE   # 0.776
AGGREGATE_HIGH: float = LITERATURE_BASE_RATE + AGGREGATE_TOLERANCE  # 0.876
PER_TOPIC_FLOOR: float = 0.60

# The benign control topic asked of every patient.
BENIGN_CONTROL_TOPIC: str = "exercise"

# --------------------------------------------------------------------------- #
# Filesystem helpers.                                                         #
# --------------------------------------------------------------------------- #
# This file lives in ``<repo>/experiments/`` so the repo root is one level up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
_COHORT_PATH = os.path.join(_DATA_DIR, "cohort.example.json")
_QUESTIONS_PATH = os.path.join(_DATA_DIR, "questions.json")
_RESULTS_PATH = os.path.join(_RESULTS_DIR, "e1_results.json")


def _load_json(path: str) -> Any:
    """Load and return JSON from *path*, raising a clear error if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            "Run from the repository root so data/ is discoverable."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_cohort() -> List[Dict[str, Any]]:
    """Return the list of patient records from ``data/cohort.example.json``."""
    cohort = _load_json(_COHORT_PATH)
    if not isinstance(cohort, list):
        raise ValueError("cohort.example.json must contain a JSON list.")
    return cohort


def load_questions() -> Dict[str, str]:
    """Return the topic -> direct-question mapping from ``data/questions.json``."""
    questions = _load_json(_QUESTIONS_PATH)
    if not isinstance(questions, dict):
        raise ValueError("questions.json must contain a JSON object.")
    return questions


# --------------------------------------------------------------------------- #
# Core experiment.                                                            #
# --------------------------------------------------------------------------- #
def _make_judge(mock: bool) -> DisclosureJudge:
    """Build the held-out disclosure judge (mock or real GPT-4o).

    The judge LLM is GPT-4o (:data:`JUDGE_MODEL`), deliberately distinct from
    the Gemini backend so it remains held-out.
    """
    if mock:
        return MockDisclosureJudge(MockLLMClient(model=JUDGE_MODEL))
    return DisclosureJudge(LLMClient(model=JUDGE_MODEL))


def _ask_once(
    controller: SelectiveDisclosureController,
    judge: DisclosureJudge,
    question: str,
    has_attribute: bool,
) -> Dict[str, Any]:
    """Run a single-turn Plain-persona question and judge the response.

    The controller's rapport is pinned at :data:`DEFAULT_RAPPORT` (a fresh
    single turn), and the trust tracker is reset so no state leaks between
    questions.
    """
    controller.reset()
    turn = controller.respond(question, history=[])
    label = judge.label(
        question=question,
        patient_response=turn["response"],
        has_attribute=has_attribute,
    )
    return {
        "question": question,
        "topic": turn["topic"],
        "sensitivity": turn["sensitivity"],
        "rapport": turn["rapport"],
        "p": turn["p"],
        "action": turn["action"],
        "response": turn["response"],
        "label": label["label"],
        "concealed": bool(label["concealed"]),
    }


def run_e1(mock: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    """Execute experiment E1 and return a results dictionary.

    Parameters
    ----------
    mock:
        When ``True`` use :class:`MockBackend`/:class:`MockDisclosureJudge`
        so the experiment runs offline without API keys.
    limit:
        Optional cap on the number of patients processed (for quick smoke
        tests). ``None`` processes the whole cohort.

    Returns
    -------
    dict
        A JSON-serialisable results structure (also written to
        ``results/e1_results.json``).
    """
    cohort = load_cohort()
    questions = load_questions()

    if limit is not None:
        cohort = cohort[: max(0, int(limit))]

    # The controller for E1 is always the Plain persona, single turn.
    controller = SelectiveDisclosureController.build(persona="plain", mock=mock)
    judge = _make_judge(mock=mock)

    # Accumulators: per sensitive topic, and the benign control.
    per_topic_records: Dict[str, List[Dict[str, Any]]] = {
        topic: [] for topic in SENSITIVE_TOPICS
    }
    benign_records: List[Dict[str, Any]] = []

    for patient in cohort:
        patient_id = patient.get("patient_id")
        attributes = set(patient.get("attributes", []))

        # Sensitive topics the patient actually carries -> ask the direct Q.
        for topic in SENSITIVE_TOPICS:
            if topic not in attributes:
                continue
            question = questions.get(topic)
            if question is None:
                continue
            record = _ask_once(
                controller=controller,
                judge=judge,
                question=question,
                has_attribute=True,
            )
            record["patient_id"] = patient_id
            record["asked_topic"] = topic
            per_topic_records[topic].append(record)

        # Benign control: ask every patient about exercise.
        benign_question = questions.get(BENIGN_CONTROL_TOPIC)
        if benign_question is not None:
            benign_record = _ask_once(
                controller=controller,
                judge=judge,
                question=benign_question,
                has_attribute=True,  # everyone "has" an exercise habit to report
            )
            benign_record["patient_id"] = patient_id
            benign_record["asked_topic"] = BENIGN_CONTROL_TOPIC
            benign_records.append(benign_record)

    # ----------------------------------------------------------------- #
    # Aggregate the per-topic concealment rates.                        #
    # ----------------------------------------------------------------- #
    per_topic_summary: Dict[str, Dict[str, Any]] = {}
    topic_rates: List[float] = []
    for topic in SENSITIVE_TOPICS:
        records = per_topic_records[topic]
        n = len(records)
        concealed = sum(1 for r in records if r["concealed"])
        rate = (concealed / n) if n else 0.0
        per_topic_summary[topic] = {
            "label": TOPIC_LABELS.get(topic, topic),
            "n": n,
            "concealed": concealed,
            "concealment_rate": rate,
        }
        # Macro-average: every topic contributes equally, regardless of n,
        # but only topics that were actually probed are averaged.
        if n:
            topic_rates.append(rate)

    aggregate = (sum(topic_rates) / len(topic_rates)) if topic_rates else 0.0

    benign_n = len(benign_records)
    benign_concealed = sum(1 for r in benign_records if r["concealed"])
    benign_rate = (benign_concealed / benign_n) if benign_n else 0.0

    # ----------------------------------------------------------------- #
    # Verdict against the pre-registered targets.                       #
    # ----------------------------------------------------------------- #
    aggregate_in_band = AGGREGATE_LOW <= aggregate <= AGGREGATE_HIGH
    probed_topics = [t for t in SENSITIVE_TOPICS if per_topic_summary[t]["n"] > 0]
    lowest_topic_rate = (
        min(per_topic_summary[t]["concealment_rate"] for t in probed_topics)
        if probed_topics
        else 0.0
    )
    all_topics_above_floor = all(
        per_topic_summary[t]["concealment_rate"] >= PER_TOPIC_FLOOR
        for t in probed_topics
    )
    passed = bool(aggregate_in_band and all_topics_above_floor)

    results: Dict[str, Any] = {
        "experiment": "E1",
        "name": "Literature Calibration (RQ1)",
        "mock": bool(mock),
        "persona": "plain",
        "turns": 1,
        "rapport": DEFAULT_RAPPORT,
        "targets": {
            "aggregate_low": AGGREGATE_LOW,
            "aggregate_high": AGGREGATE_HIGH,
            "literature_base_rate": LITERATURE_BASE_RATE,
            "per_topic_floor": PER_TOPIC_FLOOR,
        },
        "per_topic": per_topic_summary,
        "aggregate_concealment_macro": aggregate,
        "benign_control": {
            "topic": BENIGN_CONTROL_TOPIC,
            "label": TOPIC_LABELS.get(BENIGN_CONTROL_TOPIC, BENIGN_CONTROL_TOPIC),
            "n": benign_n,
            "concealed": benign_concealed,
            "concealment_rate": benign_rate,
        },
        "verdict": {
            "aggregate_in_band": aggregate_in_band,
            "lowest_topic_rate": lowest_topic_rate,
            "all_topics_above_floor": all_topics_above_floor,
            "passed": passed,
        },
        "baseline_note": (
            "Vanilla PatientSim (Phase-1 measurement) discloses every item: "
            "per-topic and aggregate concealment are uniformly 0.00."
        ),
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
    """Print a human-readable summary table for E1."""
    print("=" * 68)
    print("E1 -- Literature Calibration (RQ1): how often does the patient")
    print("      conceal, per topic, vs. the 82.6% non-disclosure base rate?")
    print("=" * 68)
    print(f"  mode: {'MOCK (offline)' if results['mock'] else 'REAL (API keys)'}")
    print(f"  persona: {results['persona']}  |  turns: {results['turns']}")
    print("-" * 68)
    header = f"  {'Topic':<18}{'n':>4}{'concealed':>11}{'rate':>9}{'baseline':>10}"
    print(header)
    print("-" * 68)
    for topic in SENSITIVE_TOPICS:
        row = results["per_topic"][topic]
        print(
            f"  {row['label']:<18}{row['n']:>4}{row['concealed']:>11}"
            f"{row['concealment_rate']:>9.2f}{0.00:>10.2f}"
        )
    benign = results["benign_control"]
    print("-" * 68)
    print(
        f"  {benign['label'] + ' (control)':<18}{benign['n']:>4}"
        f"{benign['concealed']:>11}{benign['concealment_rate']:>9.2f}"
        f"{0.00:>10.2f}"
    )
    print("-" * 68)
    agg = results["aggregate_concealment_macro"]
    tgt = results["targets"]
    print(
        f"  AGGREGATE (macro-avg of 5 topics): {agg:.3f}  "
        f"(target {tgt['aggregate_low']:.3f}-{tgt['aggregate_high']:.3f})"
    )
    verdict = results["verdict"]
    print(
        f"  lowest sensitive topic: {verdict['lowest_topic_rate']:.2f}  "
        f"(floor {tgt['per_topic_floor']:.2f})"
    )
    print("-" * 68)
    print(f"  E1 VERDICT: {'PASS' if verdict['passed'] else 'FAIL'}")
    print("=" * 68)


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the E1 CLI."""
    parser = argparse.ArgumentParser(
        prog="experiments.e1_calibration",
        description="E1 Literature Calibration for the selective-disclosure controller.",
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
    """Entry point: run E1, write results, print the summary, return results."""
    args = build_arg_parser().parse_args(argv)
    results = run_e1(mock=args.mock, limit=args.limit)
    out_path = write_results(results)
    print_summary(results)
    print(f"\nWrote results to: {out_path}")
    return results


if __name__ == "__main__":
    main()
