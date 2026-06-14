"""E3 -- Persona-Dependent Concealment (RQ3 "who").

Single-turn, paired protocol over the sensitive cohort items, repeated under
two personas:

* ``plain``        -> ``PERSONA_BIAS["plain"]``       = 0.0
* ``distrustful``  -> ``PERSONA_BIAS["distrustful"]`` = -0.7

For every (patient, topic) the patient *has*, the controller is asked the same
direct, single-turn question (rapport held at ``DEFAULT_RAPPORT``) once per
persona. A held-out disclosure judge labels each response and we mark
``partial`` or below as CONCEALED. The two personas are paired by
(patient, topic), so we can report a paired delta.

Pre-registered success criterion (Phase 2, Table 1):
    Distrustful concealment (macro-avg) > Plain concealment (macro-avg).

Reported canonical outcome (controller): Plain 0.82, Distrustful 0.94 (macro-avg),
within-pair paired delta +0.09 (95% CI +0.02 to +0.17) -> PASS. Vanilla PatientSim
baseline shows 0.00 under both personas (no persona effect -- the Phase-1 failure).

Run offline (no API keys) with::

    python -m experiments.e3_persona --mock

The ``--mock`` flag wires :class:`MockLLMClient` / :class:`MockBackend` /
:class:`MockDisclosureJudge` so the full pipeline is verifiably runnable
without any credentials.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

from selective_disclosure.config import (
    DEFAULT_RAPPORT,
    SENSITIVE_TOPICS,
    TOPIC_LABELS,
)
from selective_disclosure.controller import SelectiveDisclosureController

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")

COHORT_PATH = os.path.join(_DATA_DIR, "cohort.example.json")
QUESTIONS_PATH = os.path.join(_DATA_DIR, "questions.json")
RESULTS_PATH = os.path.join(_RESULTS_DIR, "e3_results.json")

#: Personas compared in the paired E3 protocol.
PERSONAS: List[str] = ["plain", "distrustful"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load_json(path: str) -> Any:
    """Load and return parsed JSON from ``path``."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cohort(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load the synthetic 53-patient cohort.

    Parameters
    ----------
    limit:
        Optional cap on the number of patients (useful for fast smoke runs).
    """
    cohort = _load_json(COHORT_PATH)
    if limit is not None:
        cohort = cohort[:limit]
    return cohort


def load_questions() -> Dict[str, str]:
    """Load the per-topic direct questions."""
    return _load_json(QUESTIONS_PATH)


# --------------------------------------------------------------------------- #
# Core experiment
# --------------------------------------------------------------------------- #
def _macro_average(per_topic: Dict[str, Dict[str, Any]]) -> float:
    """Macro-average a concealment rate across topics that have samples.

    Macro-averaging (rather than micro) prevents the 36 mental-illness
    patients from dominating the aggregate, matching the pre-registered
    metric definition.
    """
    rates = [
        stats["concealment_rate"]
        for stats in per_topic.values()
        if stats["n"] > 0 and stats["concealment_rate"] is not None
    ]
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def run_e3(mock: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    """Run the E3 persona-sensitivity experiment end-to-end.

    For each persona in :data:`PERSONAS`, build a controller and ask the
    single-turn direct question for every (patient, topic) the patient has.
    Returns a results dictionary (also written to ``results/e3_results.json``
    by :func:`main`).

    Parameters
    ----------
    mock:
        When ``True`` use the offline mock backend / judge (no API keys).
    limit:
        Optional cap on the number of patients.
    """
    cohort = load_cohort(limit=limit)
    questions = load_questions()

    # per_persona[persona][topic] -> {"n": int, "concealed": int}
    per_persona: Dict[str, Dict[str, Dict[str, int]]] = {
        persona: {topic: {"n": 0, "concealed": 0} for topic in SENSITIVE_TOPICS}
        for persona in PERSONAS
    }

    # Paired records keyed by (patient_id, topic) so we can compute the
    # within-pair Distrustful - Plain delta.
    paired: Dict[str, Dict[str, Any]] = {}

    # Build one controller per persona; reuse across patients/topics.
    controllers: Dict[str, SelectiveDisclosureController] = {
        persona: SelectiveDisclosureController.build(persona=persona, mock=mock)
        for persona in PERSONAS
    }

    for patient in cohort:
        patient_id = patient["patient_id"]
        attributes = patient.get("attributes", [])
        for topic in attributes:
            if topic not in SENSITIVE_TOPICS:
                continue
            question = questions.get(topic)
            if question is None:
                continue

            pair_key = f"{patient_id}:{topic}"
            paired[pair_key] = {
                "patient_id": patient_id,
                "topic": topic,
                "plain_concealed": None,
                "distrustful_concealed": None,
            }

            for persona in PERSONAS:
                controller = controllers[persona]
                # Single-turn protocol: fresh state, default rapport.
                controller.reset()
                controller.trust_tracker.rapport = DEFAULT_RAPPORT

                result = controller.respond(question, history=[])
                # controller.respond() exposes the in-line judge verdict as
                # top-level keys ("concealed"/"label"), not a nested "judge" dict.
                concealed = bool(result["concealed"])

                stats = per_persona[persona][topic]
                stats["n"] += 1
                if concealed:
                    stats["concealed"] += 1

                paired[pair_key][f"{persona}_concealed"] = concealed

    # ----- Aggregate per-topic + macro-average per persona ----------------- #
    summary: Dict[str, Any] = {"per_persona": {}}
    for persona in PERSONAS:
        per_topic_out: Dict[str, Dict[str, Any]] = {}
        for topic in SENSITIVE_TOPICS:
            stats = per_persona[persona][topic]
            n = stats["n"]
            rate = (stats["concealed"] / n) if n > 0 else None
            per_topic_out[topic] = {
                "label": TOPIC_LABELS.get(topic, topic),
                "n": n,
                "concealed": stats["concealed"],
                "concealment_rate": rate,
            }
        macro = _macro_average(per_topic_out)
        summary["per_persona"][persona] = {
            "per_topic": per_topic_out,
            "concealment_macro_avg": macro,
        }

    plain_macro = summary["per_persona"]["plain"]["concealment_macro_avg"]
    distrustful_macro = summary["per_persona"]["distrustful"]["concealment_macro_avg"]

    # ----- Paired delta over (patient, topic) ------------------------------ #
    deltas: List[float] = []
    for record in paired.values():
        if record["plain_concealed"] is None or record["distrustful_concealed"] is None:
            continue
        deltas.append(
            float(record["distrustful_concealed"]) - float(record["plain_concealed"])
        )

    n_pairs = len(deltas)
    mean_delta = (sum(deltas) / n_pairs) if n_pairs > 0 else 0.0
    ci_low, ci_high = _mean_confidence_interval(deltas)

    summary["paired"] = {
        "n_pairs": n_pairs,
        "mean_delta_distrustful_minus_plain": mean_delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }

    # ----- Verdict --------------------------------------------------------- #
    success = distrustful_macro > plain_macro
    summary["targets"] = {
        "criterion": "Distrustful concealment (macro-avg) > Plain concealment (macro-avg)",
        "plain_macro_avg": plain_macro,
        "distrustful_macro_avg": distrustful_macro,
    }
    summary["verdict"] = "PASS" if success else "FAIL"
    summary["mock"] = mock
    summary["limit"] = limit
    return summary


def _mean_confidence_interval(values: List[float], confidence: float = 0.95):
    """Return an approximate 95% CI (normal approximation) for the mean.

    Implemented with the standard library only (no SciPy dependency) to keep
    ``requirements.txt`` minimal. Uses a fixed z=1.96 for the 95% interval.
    Returns ``(None, None)`` when there are fewer than two samples.
    """
    n = len(values)
    if n < 2:
        return (None, None)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std_err = math.sqrt(variance / n)
    z = 1.96 if confidence == 0.95 else 1.96
    return (mean - z * std_err, mean + z * std_err)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_rate(rate: Optional[float]) -> str:
    """Format a concealment rate (or ``--`` when no samples)."""
    if rate is None:
        return "  --  "
    return f"{rate:6.2f}"


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable summary table for E3."""
    print("\n=== E3: Persona-Dependent Concealment (RQ3 'who') ===")
    print(f"mode: {'MOCK' if summary['mock'] else 'LIVE'}"
          f"   limit: {summary['limit']}")
    header = f"{'topic':<16}{'n':>4}{'plain':>9}{'distrust':>10}"
    print(header)
    print("-" * len(header))

    plain_topics = summary["per_persona"]["plain"]["per_topic"]
    distrust_topics = summary["per_persona"]["distrustful"]["per_topic"]
    for topic in SENSITIVE_TOPICS:
        p = plain_topics[topic]
        d = distrust_topics[topic]
        print(
            f"{p['label']:<16}{p['n']:>4}"
            f"{_fmt_rate(p['concealment_rate']):>9}"
            f"{_fmt_rate(d['concealment_rate']):>10}"
        )
    print("-" * len(header))

    plain_macro = summary["per_persona"]["plain"]["concealment_macro_avg"]
    distrust_macro = summary["per_persona"]["distrustful"]["concealment_macro_avg"]
    print(f"{'MACRO-AVG':<16}{'':>4}{plain_macro:>9.2f}{distrust_macro:>10.2f}")

    paired = summary["paired"]
    delta = paired["mean_delta_distrustful_minus_plain"]
    ci_low = paired["ci95_low"]
    ci_high = paired["ci95_high"]
    if ci_low is not None and ci_high is not None:
        ci_str = f" (95% CI {ci_low:+.3f} to {ci_high:+.3f})"
    else:
        ci_str = ""
    print(
        f"\nPaired delta (Distrustful - Plain): {delta:+.3f}{ci_str}"
        f"  over n={paired['n_pairs']} pairs"
    )
    print(f"Criterion: {summary['targets']['criterion']}")
    print(f"VERDICT: {summary['verdict']}")


def write_results(summary: Dict[str, Any], path: str = RESULTS_PATH) -> None:
    """Write the E3 results dict to ``path`` as pretty JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nWrote {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the E3 CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "E3 -- Persona-Dependent Concealment: paired Plain vs Distrustful "
            "single-turn concealment over the sensitive cohort."
        )
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run fully offline with mock backend/judge (no API keys needed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of patients (for fast smoke runs).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entry point: run E3, print the summary, and write results JSON."""
    args = build_arg_parser().parse_args(argv)
    summary = run_e3(mock=args.mock, limit=args.limit)
    print_summary(summary)
    write_results(summary)
    return summary


if __name__ == "__main__":
    main()
