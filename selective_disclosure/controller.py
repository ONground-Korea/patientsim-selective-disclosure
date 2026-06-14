"""The training-free, three-stage selective-disclosure controller.

This module wires together the three stages described in the Phase-3 method
into a single object that wraps -- but never modifies -- the PatientSim
backend:

    doctor question + history
        -> Stage 1  Sensitivity Classifier   (topic, sensitivity ``s``)
        -> Stage 2  Trust State Tracker       (rapport ``r``)
        -> Stage 3  Disclosure Policy         (p, four-way action)
        -> inject a structured instruction into the PatientSim system prompt
        -> backend (Gemini-2.5-Flash) writes the surface utterance

The backend is held fixed (Gemini-2.5-Flash, CEFR=C, Recall=High,
Dazed=Normal); no fine-tuning is performed.  All control is exercised purely
through the injected structured instruction, which is why the design is
"training-free".

The :class:`SelectiveDisclosureController.build` classmethod is the convenient
entry point used by the experiments: it constructs the LLM client(s) (real or
mock), all three stages, and the judge, and returns a ready controller.  In
``mock=True`` mode the whole pipeline is deterministic and runs offline with
no API keys.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import DEFAULT_RAPPORT
from .disclosure_policy import (
    action_to_instruction,
    decide_action,
    disclosure_probability,
    is_concealed,
)
from .sensitivity_classifier import SensitivityClassifier
from .trust_tracker import TrustTracker

__all__ = ["SelectiveDisclosureController"]


class SelectiveDisclosureController:
    """Wrap PatientSim with topic-, rapport-, and persona-gated disclosure.

    Parameters
    ----------
    backend:
        A PatientSim backend exposing ``generate(instruction, history) -> str``
        (:class:`~selective_disclosure.backends.PatientSimBackend` or
        :class:`~selective_disclosure.backends.MockBackend`). The controller
        only ever talks to the backend through the injected structured
        instruction; it never touches the backend's own model settings.
    classifier:
        Stage 1 -- a :class:`~selective_disclosure.sensitivity_classifier.SensitivityClassifier`.
    trust_tracker:
        Stage 2 -- a :class:`~selective_disclosure.trust_tracker.TrustTracker`
        holding the running rapport state.
    judge:
        Optional held-out disclosure judge
        (:class:`~selective_disclosure.judge.DisclosureJudge`). When supplied,
        :meth:`respond` will also attach a judged ``label`` / ``concealed``
        verdict to its result so callers do not need a second pass.
    persona:
        The patient persona key (``"open"``/``"plain"``/``"anxious"``/
        ``"distrustful"``) used by Stage 3's ``persona_bias`` term and passed
        through to the backend.
    """

    def __init__(
        self,
        backend,
        classifier: SensitivityClassifier,
        trust_tracker: TrustTracker,
        judge=None,
        persona: str = "plain",
    ) -> None:
        self.backend = backend
        self.classifier = classifier
        self.trust_tracker = trust_tracker
        self.judge = judge
        self.persona = (persona or "plain").strip().lower()
        # Keep the backend's persona in sync so the surface utterance reflects
        # the same patient we are gating for.
        if hasattr(self.backend, "persona"):
            self.backend.persona = self.persona

    # ------------------------------------------------------------------
    # Core turn
    # ------------------------------------------------------------------
    def respond(self, doctor_utterance: str, history: Optional[List[dict]] = None) -> Dict:
        """Run one full controller turn and return the structured result.

        The pipeline is executed in order: classify the topic/sensitivity,
        update the rapport state from the doctor's utterance, compute the
        disclosure probability and four-way action, build the structured
        instruction, and finally ask the backend to generate the patient's
        surface utterance.

        Parameters
        ----------
        doctor_utterance:
            The clinician's question / statement for this turn.
        history:
            Prior conversation turns as ``{"role": ..., "content": ...}`` dicts.
            Defaults to an empty history.

        Returns
        -------
        dict
            Keys: ``topic``, ``sensitivity``, ``rapport``, ``p``, ``action``,
            ``instruction``, ``response``. If a judge was supplied, the dict
            also carries ``label`` and ``concealed``.
        """
        history = list(history) if history else []

        # Stage 1 -- sensitivity classification.
        classification = self.classifier.classify(doctor_utterance)
        topic = classification["topic"]
        sensitivity = float(classification["sensitivity"])

        # Stage 2 -- update the running rapport from this doctor utterance.
        rapport = float(self.trust_tracker.update(doctor_utterance))

        # Stage 3 -- disclosure probability and four-way action.
        p = disclosure_probability(
            sensitivity=sensitivity,
            rapport=rapport,
            persona=self.persona,
        )
        action = decide_action(p)

        # Build the structured instruction that is injected into PatientSim.
        reason = (
            f"topic={topic}, s={sensitivity:.2f}, r={rapport:.2f}, "
            f"persona={self.persona}, p={p:.3f}"
        )
        instruction = action_to_instruction(
            topic=topic,
            action=action,
            sensitivity=sensitivity,
            rapport=rapport,
            persona=self.persona,
            reason=reason,
        )

        # Backend generates the surface utterance from the injected instruction.
        response = self.backend.generate(instruction, history)

        result: Dict = {
            "topic": topic,
            "sensitivity": sensitivity,
            "rapport": rapport,
            "p": p,
            "action": action,
            "instruction": instruction,
            "response": response,
        }

        # Optional in-line judging.
        if self.judge is not None:
            has_attribute = self._infer_has_attribute(history, topic)
            verdict = self.judge.label(doctor_utterance, response, has_attribute)
            result["label"] = verdict.get("label")
            result["concealed"] = bool(
                verdict.get("concealed", is_concealed(verdict.get("label", action)))
            )

        return result

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the rapport state back to its initial value.

        Sensitivity classification and the disclosure policy are stateless, so
        only the Stage-2 trust tracker needs resetting between sessions /
        patients.
        """
        self.trust_tracker.reset()

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        persona: str = "plain",
        mock: bool = False,
        patient_record: Optional[dict] = None,
    ) -> "SelectiveDisclosureController":
        """Wire up every stage and return a ready-to-use controller.

        Parameters
        ----------
        persona:
            Patient persona key for Stage 3 and the backend.
        mock:
            When ``True`` use :class:`~selective_disclosure.llm.MockLLMClient`,
            :class:`~selective_disclosure.backends.MockBackend`, and
            :class:`~selective_disclosure.judge.MockDisclosureJudge` so the
            controller runs deterministically offline with no API keys. When
            ``False`` use the real REST clients and the PatientSim backend
            (which read keys from environment variables).
        patient_record:
            Optional patient record forwarded to the backend (used by
            PatientSim to ground the surface utterance).

        Returns
        -------
        SelectiveDisclosureController
            A fully wired controller.
        """
        # Imported lazily so the lightweight model classes can be imported
        # without pulling the backend / judge modules unless ``build`` is used.
        from .backends import MockBackend, PatientSimBackend
        from .config import BACKEND_MODEL, DOCTOR_MODEL, JUDGE_MODEL
        from .judge import DisclosureJudge, MockDisclosureJudge
        from .llm import LLMClient, MockLLMClient

        if mock:
            backend_llm = MockLLMClient(model=BACKEND_MODEL)
            judge_llm = MockLLMClient(model=JUDGE_MODEL)
            classifier_llm = MockLLMClient(model=JUDGE_MODEL)
            tracker_llm = MockLLMClient(model=JUDGE_MODEL)
            backend = MockBackend(
                llm=backend_llm, persona=persona, patient_record=patient_record
            )
            judge = MockDisclosureJudge(judge_llm)
        else:
            backend_llm = LLMClient(model=BACKEND_MODEL)
            judge_llm = LLMClient(model=JUDGE_MODEL)
            classifier_llm = LLMClient(model=JUDGE_MODEL)
            tracker_llm = LLMClient(model=JUDGE_MODEL)
            backend = PatientSimBackend(
                llm=backend_llm, persona=persona, patient_record=patient_record
            )
            judge = DisclosureJudge(judge_llm)

        classifier = SensitivityClassifier(classifier_llm)
        trust_tracker = TrustTracker(tracker_llm, init_rapport=DEFAULT_RAPPORT)

        return cls(
            backend=backend,
            classifier=classifier,
            trust_tracker=trust_tracker,
            judge=judge,
            persona=persona,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_has_attribute(self, history: List[dict], topic: str) -> bool:
        """Best-effort guess of whether the simulated patient holds ``topic``.

        The authoritative source is the backend's ``patient_record`` (its
        ``attributes`` list). If that is unavailable, fall back to ``True`` so
        that a concealment is judged against the assumption the patient does
        carry the attribute (the experiments always query patients on topics
        they hold).
        """
        record = getattr(self.backend, "patient_record", None)
        if isinstance(record, dict):
            attrs = record.get("attributes")
            if isinstance(attrs, (list, tuple, set)):
                return topic in attrs
        return True
