"""run_all -- run E1, E2, E3 and print an overall PASS/FAIL summary.

This driver executes the three pre-registered experiments end-to-end and
checks each against its Phase-2 success criterion (Table 1):

    E1  aggregate concealment within 82.6% +/- 5% (i.e. 77.6%-87.6%)
        AND every sensitive topic >= 60%.
    E2  trust-sensitivity slope(sensitive) > 0
        AND slope(sensitive) > slope(benign control).
    E3  Distrustful concealment (macro-avg) > Plain concealment (macro-avg).

Each experiment writes its own ``results/eN_results.json``; this script
aggregates the verdicts and exits non-zero if any experiment fails (handy for
CI). Run a fully offline smoke test with::

    python -m experiments.run_all --mock

or, with real API keys configured in the environment::

    python -m experiments.run_all

No credentials are ever read or stored by this module; key handling lives
entirely in :class:`selective_disclosure.llm.LLMClient`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from experiments.e1_calibration import run_e1
from experiments.e2_rapport import run_e2
from experiments.e3_persona import run_e3

# --------------------------------------------------------------------------- #
# Pre-registered targets (Phase 2, Table 1)
# --------------------------------------------------------------------------- #
E1_TARGET_CENTER = 0.826
E1_TARGET_TOLERANCE = 0.05
E1_TARGET_LOW = E1_TARGET_CENTER - E1_TARGET_TOLERANCE   # 0.776
E1_TARGET_HIGH = E1_TARGET_CENTER + E1_TARGET_TOLERANCE  # 0.876
E1_PER_TOPIC_FLOOR = 0.60


# --------------------------------------------------------------------------- #
# Verdict extraction helpers
#
# Each experiment already computes its own verdict, but we re-derive the
# boolean here so the overall summary is self-contained and does not silently
# trust a possibly-missing "verdict" string.
# --------------------------------------------------------------------------- #
def _e1_pass(summary: Dict[str, Any]) -> bool:
    """Whether the E1 result dict satisfies the pre-registered criterion."""
    aggregate = summary.get("aggregate_concealment_macro")
    if aggregate is None:
        aggregate = summary.get("aggregate_concealment")
    if aggregate is None:
        aggregate = summary.get("aggregate", {}).get("concealment_macro_avg")
    if aggregate is None:
        return _verdict_is_pass(summary)
    within_band = E1_TARGET_LOW <= aggregate <= E1_TARGET_HIGH

    per_topic = summary.get("per_topic", {})
    floors_ok = True
    for stats in per_topic.values():
        rate = stats.get("concealment_rate")
        if rate is not None and rate < E1_PER_TOPIC_FLOOR:
            floors_ok = False
            break
    return bool(within_band and floors_ok)


def _e2_pass(summary: Dict[str, Any]) -> bool:
    """Whether the E2 result dict satisfies the pre-registered criterion."""
    sensitive = _get_slope(summary, "sensitive")
    benign = _get_slope(summary, "benign")
    if sensitive is None:
        return _verdict_is_pass(summary)
    benign_val = benign if benign is not None else 0.0
    return bool(sensitive > 0 and sensitive > benign_val)


def _e3_pass(summary: Dict[str, Any]) -> bool:
    """Whether the E3 result dict satisfies the pre-registered criterion."""
    per_persona = summary.get("per_persona", {})
    plain = per_persona.get("plain", {}).get("concealment_macro_avg")
    distrustful = per_persona.get("distrustful", {}).get("concealment_macro_avg")
    if plain is None or distrustful is None:
        return _verdict_is_pass(summary)
    return bool(distrustful > plain)


def _get_slope(summary: Dict[str, Any], kind: str) -> Optional[float]:
    """Best-effort extraction of a slope value from an E2 result dict.

    Tolerates a few plausible key layouts produced by ``e2_rapport`` so this
    driver stays robust to minor shape differences.
    """
    # Flat keys, e.g. {"slope_sensitive": 0.41, "slope_benign": 0.06}.
    for key in (f"slope_{kind}", f"{kind}_slope"):
        if key in summary and summary[key] is not None:
            return float(summary[key])
    # Nested under "slopes".
    slopes = summary.get("slopes", {})
    if kind in slopes and slopes[kind] is not None:
        return float(slopes[kind])
    # Nested under "summary"/"aggregate".
    for container_key in ("summary", "aggregate"):
        container = summary.get(container_key, {})
        if isinstance(container, dict):
            for key in (f"slope_{kind}", f"{kind}_slope", kind):
                value = container.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def _verdict_is_pass(summary: Dict[str, Any]) -> bool:
    """Fallback: trust the experiment's own ``verdict`` if present.

    E1/E2 store ``verdict`` as a dict with a ``passed`` bool; E3 stores it as a
    ``"PASS"``/``"FAIL"`` string. Handle both shapes.
    """
    verdict = summary.get("verdict")
    if isinstance(verdict, dict):
        return bool(verdict.get("passed"))
    return str(verdict or "").strip().upper() == "PASS"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_all(mock: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    """Run E1, E2, E3 in order and collect their result dicts + verdicts.

    Parameters
    ----------
    mock:
        When ``True`` every experiment runs fully offline (mock backend/judge).
    limit:
        Optional per-experiment cap on the number of patients.

    Returns
    -------
    dict
        ``{"experiments": {...}, "verdicts": {...}, "overall": "PASS"|"FAIL"}``.
    """
    print("=" * 64)
    print(" Running E1 -> E2 -> E3 "
          f"({'MOCK' if mock else 'LIVE'} mode, limit={limit})")
    print("=" * 64)

    e1 = run_e1(mock=mock, limit=limit)
    e2 = run_e2(mock=mock, limit=limit)
    e3 = run_e3(mock=mock, limit=limit)

    verdicts = {
        "E1": _e1_pass(e1),
        "E2": _e2_pass(e2),
        "E3": _e3_pass(e3),
    }
    overall = "PASS" if all(verdicts.values()) else "FAIL"

    return {
        "experiments": {"E1": e1, "E2": e2, "E3": e3},
        "verdicts": verdicts,
        "overall": overall,
        "mock": mock,
        "limit": limit,
    }


def print_overall_summary(report: Dict[str, Any]) -> None:
    """Print the consolidated PASS/FAIL summary against pre-registered targets."""
    e1 = report["experiments"]["E1"]
    e2 = report["experiments"]["E2"]
    e3 = report["experiments"]["E3"]
    verdicts = report["verdicts"]

    aggregate = e1.get("aggregate_concealment_macro")
    if aggregate is None:
        aggregate = e1.get("aggregate_concealment")
    if aggregate is None:
        aggregate = e1.get("aggregate", {}).get("concealment_macro_avg")
    e1_sensitive_floor = _min_sensitive_rate(e1)

    e2_sensitive = _get_slope(e2, "sensitive")
    e2_benign = _get_slope(e2, "benign")

    e3_plain = (
        e3.get("per_persona", {}).get("plain", {}).get("concealment_macro_avg")
    )
    e3_distrust = (
        e3.get("per_persona", {}).get("distrustful", {}).get("concealment_macro_avg")
    )

    print("\n" + "=" * 64)
    print(" OVERALL SUMMARY vs PRE-REGISTERED TARGETS (Table 1)")
    print("=" * 64)

    rows: List[str] = []
    rows.append(
        f"E1  aggregate={_fmt(aggregate)} "
        f"(target {E1_TARGET_LOW:.3f}-{E1_TARGET_HIGH:.3f}); "
        f"min topic={_fmt(e1_sensitive_floor)} (floor {E1_PER_TOPIC_FLOOR:.2f})"
        f"   -> {_badge(verdicts['E1'])}"
    )
    rows.append(
        f"E2  slope(sensitive)={_fmt(e2_sensitive)} > 0 "
        f"and > slope(benign)={_fmt(e2_benign)}"
        f"   -> {_badge(verdicts['E2'])}"
    )
    rows.append(
        f"E3  distrustful={_fmt(e3_distrust)} > plain={_fmt(e3_plain)}"
        f"   -> {_badge(verdicts['E3'])}"
    )
    for row in rows:
        print(row)

    print("-" * 64)
    print(f" OVERALL: {report['overall']}")
    print("=" * 64)


def _min_sensitive_rate(e1_summary: Dict[str, Any]) -> Optional[float]:
    """Return the lowest per-topic concealment rate reported by E1."""
    per_topic = e1_summary.get("per_topic", {})
    rates = [
        stats.get("concealment_rate")
        for stats in per_topic.values()
        if isinstance(stats, dict) and stats.get("concealment_rate") is not None
    ]
    return min(rates) if rates else None


def _fmt(value: Optional[float]) -> str:
    """Format an optional float for the summary table."""
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if value < 0 else f"{value:.3f}"


def _badge(passed: bool) -> str:
    """Return a PASS/FAIL badge string."""
    return "PASS" if passed else "FAIL"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the run-all CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all three pre-registered experiments (E1, E2, E3) and print "
            "an overall PASS/FAIL summary against the Phase-2 targets."
        )
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run every experiment fully offline (no API keys needed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of patients per experiment (fast smoke run).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = all PASS)."""
    args = build_arg_parser().parse_args(argv)
    report = run_all(mock=args.mock, limit=args.limit)
    print_overall_summary(report)
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
