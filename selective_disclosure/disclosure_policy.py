"""Stage 3 of the training-free controller: the Disclosure Policy.

This module is pure (no LLM calls). Given the topic sensitivity ``s`` (Stage 1),
the accumulated rapport ``r`` (Stage 2), and the patient persona, it computes a
closed-form probability of disclosure and maps it to one of four discrete
actions. The chosen action is then rendered as a structured instruction block
that is injected into the PatientSim system prompt; the backend
(Gemini-2.5-Flash) writes the surface utterance.

Disclosure probability (sigmoid)::

    p(disclose) = sigmoid( b + w_p * persona_bias + w_s * (1 - s) + w_r * r )

with fitted weights ``b = -1.3``, ``w_p = 1.5``, ``w_s = 3.2``, ``w_r = 2.6``
(see :data:`selective_disclosure.config.POLICY_WEIGHTS`). Note the
``(1 - s)`` term: the *more* sensitive a topic, the *lower* the disclosure
probability, so highly sensitive items are concealed unless rapport is high.

Four-way action by fixed thresholds on ``p`` (see
:data:`selective_disclosure.config.DISCLOSURE_THRESHOLDS`):

* ``p > 0.85``         -> ``"full"``          (e.g. "Yes, I have HIV.")
* ``0.55 <= p <= 0.85`` -> ``"partial"``       (e.g. "I have some health issues.")
* ``0.25 <= p < 0.55``  -> ``"evasive"``       (e.g. "Why does that matter?")
* ``p < 0.25``         -> ``"false_denial"``  (e.g. "No.")

Every action except ``"full"`` counts as CONCEALED for evaluation purposes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .config import DISCLOSURE_THRESHOLDS, PERSONA_BIAS, POLICY_WEIGHTS

__all__ = [
    "ACTION_TONES",
    "sigmoid",
    "disclosure_probability",
    "decide_action",
    "is_concealed",
    "action_to_instruction",
]

# Canonical action labels emitted by :func:`decide_action`.
FULL = "full"
PARTIAL = "partial"
EVASIVE = "evasive"
FALSE_DENIAL = "false_denial"

# Tone guidance handed to the backend for each action. These describe *how*
# the patient should phrase the chosen disclosure decision, not the content.
ACTION_TONES: Dict[str, str] = {
    FULL: (
        "Answer honestly and directly. Acknowledge the condition or behaviour "
        "plainly, as a patient who has decided to trust the clinician."
    ),
    PARTIAL: (
        "Disclose only vaguely. Hint that there is something relevant without "
        "naming the specific sensitive condition; stay guarded and minimal."
    ),
    EVASIVE: (
        "Deflect without answering. Question why it is being asked, change the "
        "subject, or give a non-committal response; do not confirm or deny."
    ),
    FALSE_DENIAL: (
        "Deny outright. Flatly state that this does not apply to you, as a "
        "patient too ashamed or distrustful to admit it."
    ),
}


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid ``1 / (1 + exp(-x))``.

    Args:
        x: The logit.

    Returns:
        A probability in the open interval ``(0, 1)``.
    """
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def disclosure_probability(
    sensitivity: float,
    rapport: float,
    persona: str,
    weights: Optional[Dict[str, float]] = None,
    persona_bias: Optional[Dict[str, float]] = None,
) -> float:
    """Compute the closed-form probability that the patient discloses.

    Implements ``sigmoid(b + w_p * persona_bias + w_s * (1 - s) + w_r * r)``.

    Args:
        sensitivity: Topic sensitivity ``s`` in ``[0, 1]`` from Stage 1.
        rapport: Accumulated rapport ``r`` in ``[0, 1]`` from Stage 2.
        persona: Persona key (e.g. ``"plain"``, ``"distrustful"``). Resolved
            against ``persona_bias``; unknown personas contribute ``0.0`` bias.
        weights: Optional override for the policy weights. When ``None`` the
            fitted :data:`POLICY_WEIGHTS` are used. Expected keys:
            ``"intercept"``, ``"w_persona"``, ``"w_s"``, ``"w_r"``.
        persona_bias: Optional override for the persona-bias table. When
            ``None`` the canonical :data:`PERSONA_BIAS` is used.

    Returns:
        Probability of disclosure in ``(0, 1)``.
    """
    w = POLICY_WEIGHTS if weights is None else weights
    biases = PERSONA_BIAS if persona_bias is None else persona_bias

    bias = biases.get(persona, 0.0)
    logit = (
        w["intercept"]
        + w["w_persona"] * bias
        + w["w_s"] * (1.0 - sensitivity)
        + w["w_r"] * rapport
    )
    return sigmoid(logit)


def decide_action(p: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """Map a disclosure probability to a discrete action.

    Threshold semantics (matching the contract):

    * ``p > full``            -> ``"full"``
    * ``partial <= p <= full`` -> ``"partial"``
    * ``evasive <= p < partial`` -> ``"evasive"``
    * ``p < evasive``         -> ``"false_denial"``

    Args:
        p: Disclosure probability in ``[0, 1]``.
        thresholds: Optional override mapping with keys ``"full"``,
            ``"partial"``, ``"evasive"``. When ``None`` the canonical
            :data:`DISCLOSURE_THRESHOLDS` are used.

    Returns:
        One of ``"full"``, ``"partial"``, ``"evasive"``, ``"false_denial"``.
    """
    t = DISCLOSURE_THRESHOLDS if thresholds is None else thresholds
    if p > t["full"]:
        return FULL
    if p >= t["partial"]:
        return PARTIAL
    if p >= t["evasive"]:
        return EVASIVE
    return FALSE_DENIAL


def is_concealed(action_or_label: str) -> bool:
    """Return whether an action/label counts as concealment.

    Any decision other than a full, honest disclosure is concealment. The
    comparison is case-insensitive and tolerant of surrounding whitespace.

    Args:
        action_or_label: An action from :func:`decide_action` or a judge label.

    Returns:
        ``False`` only when the action/label is ``"full"``; ``True`` otherwise.
    """
    return str(action_or_label).strip().lower() != FULL


def action_to_instruction(
    topic: str,
    action: str,
    sensitivity: float,
    rapport: float,
    persona: str,
    reason: str = "",
) -> str:
    """Render a structured instruction block for the PatientSim backend.

    The returned text is injected into PatientSim's system prompt. It tells the
    backend *what decision* the controller made (topic, action, tone) and the
    contextual signals behind it, while leaving the actual wording of the
    patient utterance to the backend (Gemini-2.5-Flash).

    Args:
        topic: Topic key the doctor's question concerns (e.g. ``"hiv"``).
        action: Chosen action from :func:`decide_action`.
        sensitivity: Topic sensitivity ``s`` in ``[0, 1]``.
        rapport: Current rapport ``r`` in ``[0, 1]``.
        persona: Active persona key.
        reason: Optional free-text rationale to include for traceability.

    Returns:
        A multi-line, human-readable instruction block.
    """
    normalized = str(action).strip().lower()
    tone = ACTION_TONES.get(normalized, ACTION_TONES[EVASIVE])
    reason_line = reason.strip() if reason and reason.strip() else "(not specified)"

    lines = [
        "[DISCLOSURE CONTROL INSTRUCTION]",
        f"topic: {topic}",
        f"persona: {persona}",
        f"sensitivity (s): {sensitivity:.2f}",
        f"rapport (r): {rapport:.2f}",
        f"decision: {normalized}",
        f"tone: {tone}",
        f"reason: {reason_line}",
        (
            "Stay fully in character as this patient. Express ONLY the decision "
            "above in your own natural words; do not reveal that an instruction "
            "was given and do not mention this control block."
        ),
    ]
    return "\n".join(lines)
