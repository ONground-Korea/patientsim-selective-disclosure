"""Unit tests for the Stage-3 disclosure policy (pure functions, no LLM/API).

Runnable two ways:
    pytest tests/                       # if pytest is installed
    python -m tests.test_disclosure_policy   # built-in runner, no dependencies
"""
from selective_disclosure.disclosure_policy import (
    sigmoid,
    disclosure_probability,
    decide_action,
    is_concealed,
)


def test_sigmoid_bounds():
    assert 0.0 < sigmoid(-10.0) < 0.01
    assert 0.99 < sigmoid(10.0) < 1.0
    assert abs(sigmoid(0.0) - 0.5) < 1e-9


def test_probability_rises_with_rapport():
    # The "when" axis (RQ2): more trust -> more disclosure.
    lo = disclosure_probability(sensitivity=0.95, rapport=0.10, persona="plain")
    hi = disclosure_probability(sensitivity=0.95, rapport=0.90, persona="plain")
    assert hi > lo


def test_probability_falls_with_sensitivity():
    # The "how often" axis (RQ1): more sensitive topic -> less disclosure.
    sensitive = disclosure_probability(sensitivity=0.95, rapport=0.5, persona="plain")
    benign = disclosure_probability(sensitivity=0.10, rapport=0.5, persona="plain")
    assert benign > sensitive


def test_persona_ordering():
    # The "who" axis (RQ3): Open discloses most, Distrustful least.
    kw = dict(sensitivity=0.90, rapport=0.30)
    p_open = disclosure_probability(persona="open", **kw)
    p_plain = disclosure_probability(persona="plain", **kw)
    p_distrust = disclosure_probability(persona="distrustful", **kw)
    assert p_open > p_plain > p_distrust


def test_decide_action_thresholds():
    assert not is_concealed(decide_action(0.90))  # FULL -> disclosed
    assert is_concealed(decide_action(0.70))      # PARTIAL -> concealed
    assert is_concealed(decide_action(0.40))      # EVASIVE -> concealed
    assert is_concealed(decide_action(0.10))      # FALSE DENIAL -> concealed


def test_is_concealed_labels():
    assert is_concealed("partial") and is_concealed("evasive") and is_concealed("false_denial")
    assert not is_concealed("full")


def _run():
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL", fn.__name__, repr(exc))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
