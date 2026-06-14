"""End-to-end wiring tests for the controller in offline mock mode (no API keys).

Runnable two ways:
    pytest tests/                    # if pytest is installed
    python -m tests.test_controller  # built-in runner, no dependencies
"""
from selective_disclosure import SelectiveDisclosureController
from selective_disclosure.disclosure_policy import is_concealed


def _respond(persona, question="Do you have HIV?", rapport=0.10):
    controller = SelectiveDisclosureController.build(persona=persona, mock=True)
    controller.trust_tracker.rapport = rapport
    return controller.respond(question, history=[])


def test_respond_has_expected_keys():
    result = _respond("plain")
    for key in ("topic", "sensitivity", "rapport", "p", "action",
                "instruction", "response", "concealed", "label"):
        assert key in result, "missing key: " + key


def test_sensitive_topic_concealed_at_low_rapport():
    # A highly sensitive topic with no rapport built should not be fully disclosed.
    result = _respond("plain")
    assert result["sensitivity"] >= 0.80
    assert is_concealed(result["action"])


def test_distrustful_discloses_no_more_than_plain():
    # Persona axis (RQ3): distrustful -> lower disclosure probability.
    p_plain = _respond("plain")["p"]
    p_distrust = _respond("distrustful")["p"]
    assert p_distrust <= p_plain


def test_benign_topic_disclosed_under_plain():
    # Specificity: a benign topic should be fully disclosed even with low rapport.
    result = _respond("plain", question="Do you exercise regularly?")
    assert result["sensitivity"] <= 0.30
    assert not is_concealed(result["action"])


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
