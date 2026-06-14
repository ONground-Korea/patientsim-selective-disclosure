"""Stage 2 of the training-free controller: the Trust State Tracker.

Vanilla PatientSim has no notion of accumulated rapport, so its turn-1 and
turn-20 behaviour are indistinguishable. This module adds a lightweight,
scalar trust state ``r in [0, 1]`` that is updated once per doctor utterance.

Each turn an LLM-judge scores the doctor's utterance on four cues -- empathy,
confidentiality framing, validation, and hostility -- each in ``[0, 1]``. The
scalar rapport is then updated with the clipped linear rule (weights live in
:data:`selective_disclosure.config.RAPPORT_WEIGHTS`)::

    r_new = clip(
        r_old
        + 0.15 * empathy
        + 0.20 * confidentiality
        + 0.10 * validation
        - 0.25 * hostility,
        0, 1,
    )

The resulting rapport scalar feeds Stage 3 (:mod:`disclosure_policy`) through
the ``w_r * r`` term of the disclosure-probability sigmoid.
"""

from __future__ import annotations

from typing import Dict, Optional

from .config import DEFAULT_RAPPORT, RAPPORT_WEIGHTS

__all__ = ["clip", "TrustTracker"]

# The four cues the LLM-judge is asked to score, in [0, 1].
_CUES = ("empathy", "confidentiality", "validation", "hostility")

# System prompt for the LLM-judge that scores a single doctor utterance.
_TRUST_JUDGE_SYSTEM = (
    "You are an expert in clinical communication. You will be shown a single "
    "utterance spoken by a doctor to a patient during a medical interview. "
    "Rate the utterance on four independent cues, each on a continuous scale "
    "from 0.0 (cue completely absent) to 1.0 (cue strongly present):\n"
    "- empathy: warmth, compassion, acknowledgement of the patient's feelings.\n"
    "- confidentiality: explicit assurance that what is shared stays private "
    "and is used only to help the patient.\n"
    "- validation: normalising the patient's experience, reducing shame or "
    "judgement.\n"
    "- hostility: coldness, pressure, accusation, impatience, or dismissiveness.\n"
    "Respond with ONLY a JSON object of the form "
    '{"empathy": float, "confidentiality": float, "validation": float, '
    '"hostility": float}.'
)


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into the closed interval ``[low, high]``.

    Args:
        value: The number to clamp.
        low: Lower bound (inclusive). Defaults to ``0.0``.
        high: Upper bound (inclusive). Defaults to ``1.0``.

    Returns:
        ``value`` constrained to ``[low, high]``.
    """
    return max(low, min(high, value))


class TrustTracker:
    """Maintain a running rapport scalar across an interview.

    The tracker holds a single mutable scalar :attr:`rapport` in ``[0, 1]``.
    On every doctor utterance, :meth:`update` asks the supplied LLM to score
    the four communication cues and applies the clipped linear update rule.

    Attributes:
        llm: An object exposing ``complete_json(system, user) -> dict`` (e.g.
            :class:`selective_disclosure.llm.LLMClient` or its mock).
        weights: Mapping of cue name -> signed weight. Defaults to
            :data:`selective_disclosure.config.RAPPORT_WEIGHTS`.
        init_rapport: The rapport value restored on construction and on
            :meth:`reset`.
        rapport: The current rapport scalar in ``[0, 1]``.
    """

    def __init__(
        self,
        llm,
        weights: Optional[Dict[str, float]] = None,
        init_rapport: float = DEFAULT_RAPPORT,
    ) -> None:
        """Initialise the tracker.

        Args:
            llm: LLM client exposing ``complete_json(system, user) -> dict``.
            weights: Optional override for the cue weights. When ``None`` the
                canonical :data:`RAPPORT_WEIGHTS` are used. A defensive copy is
                stored so callers cannot mutate shared module state.
            init_rapport: Starting rapport, clamped into ``[0, 1]``. Also the
                value restored by :meth:`reset`.
        """
        self.llm = llm
        self.weights: Dict[str, float] = dict(
            RAPPORT_WEIGHTS if weights is None else weights
        )
        self.init_rapport: float = clip(float(init_rapport))
        self.rapport: float = self.init_rapport

    def _score_cues(self, doctor_utterance: str) -> Dict[str, float]:
        """Ask the LLM-judge to score the four cues for one utterance.

        Robust to missing or malformed keys: any cue the model omits is
        treated as ``0.0`` and every returned value is clamped to ``[0, 1]``.

        Args:
            doctor_utterance: The doctor's single turn of dialogue.

        Returns:
            Mapping of each cue name in :data:`_CUES` to a float in ``[0, 1]``.
        """
        user = f"Doctor utterance:\n{doctor_utterance}"
        raw = self.llm.complete_json(_TRUST_JUDGE_SYSTEM, user)
        if not isinstance(raw, dict):
            raw = {}

        scores: Dict[str, float] = {}
        for cue in _CUES:
            try:
                scores[cue] = clip(float(raw.get(cue, 0.0)))
            except (TypeError, ValueError):
                scores[cue] = 0.0
        return scores

    def update(self, doctor_utterance: str) -> float:
        """Score a doctor utterance and advance the rapport state.

        Applies the clipped linear update rule using the configured weights
        and stores the result on :attr:`rapport`.

        Args:
            doctor_utterance: The doctor's single turn of dialogue.

        Returns:
            The updated rapport scalar in ``[0, 1]``.
        """
        scores = self._score_cues(doctor_utterance)
        delta = sum(self.weights.get(cue, 0.0) * scores[cue] for cue in _CUES)
        self.rapport = clip(self.rapport + delta)
        return self.rapport

    def reset(self) -> None:
        """Restore the rapport state to :attr:`init_rapport`.

        Call this between independent interview sessions so that trust built
        in one session does not leak into the next.
        """
        self.rapport = self.init_rapport
