"""Scripted doctor agents used by the rapport experiment (E2).

This module provides two contrasting clinician styles that drive a
multi-turn history-taking interview against the :class:`PatientSim` backend
(wrapped by :class:`~selective_disclosure.controller.SelectiveDisclosureController`):

* **Empathetic** -- empathic acknowledgement, explicit confidentiality
  framing, validation, and gradual probing towards the target topic.
* **Cold** -- direct closed questions, no reassurance or rapport building.

The two styles are the experimental manipulation in E2 (RQ2, "when"):
an empathetic clinician should grow the patient's rapport ``r`` and therefore
the disclosure probability, while a cold clinician should not.  The benign
control topic (``exercise``) is handled with the same machinery so that a
system which merely "gets chattier everywhere" does not spuriously pass.

The agents are deliberately *style*-driven rather than content-driven: they
are told which ``target_topic`` they are working towards and asked to produce
the next single utterance given the running conversation history.  A
:class:`~selective_disclosure.llm.MockLLMClient` yields deterministic
utterances so the whole experiment runs offline without API keys.
"""

from __future__ import annotations

from typing import Dict, List

from .config import TOPIC_LABELS

__all__ = ["EMPATHETIC_SYSTEM", "COLD_SYSTEM", "DoctorAgent"]


# ---------------------------------------------------------------------------
# Style system prompts
# ---------------------------------------------------------------------------

EMPATHETIC_SYSTEM: str = (
    "You are a warm, experienced physician taking a medical history from a "
    "patient. Your interviewing style is EMPATHETIC:\n"
    "- Open with empathic acknowledgement of how the patient is feeling.\n"
    "- Explicitly reassure the patient about confidentiality (\"everything you "
    "tell me stays between us\").\n"
    "- Validate the patient's concerns before asking anything sensitive.\n"
    "- Probe GRADUALLY: build rapport first, then ease towards the sensitive "
    "topic with gentle, open-ended questions.\n"
    "- Never sound hostile, rushed, or judgemental.\n"
    "Produce ONLY the doctor's next single spoken utterance (1-2 sentences). "
    "Do not include stage directions, names, or quotation marks."
)

COLD_SYSTEM: str = (
    "You are a brusque, time-pressured physician taking a medical history. "
    "Your interviewing style is COLD:\n"
    "- Ask direct, closed questions that can be answered yes/no.\n"
    "- Offer NO reassurance, empathy, or confidentiality framing.\n"
    "- Do not validate the patient's feelings; stay clinical and detached.\n"
    "- Move quickly and bluntly toward the target topic.\n"
    "Produce ONLY the doctor's next single spoken utterance (1-2 sentences). "
    "Do not include stage directions, names, or quotation marks."
)


# Per-style canned opening / follow-up phrasing keyed by topic.  These give the
# mock client (and a missing-key fallback) sensible, deterministic behaviour
# without ever calling a network endpoint.
_EMPATHETIC_TURNS: Dict[str, List[str]] = {
    "_default": [
        "Thank you for coming in today. I can see this isn't easy, and I want "
        "you to know that everything we discuss stays completely confidential.",
        "You're in a safe space here, and there's no judgement at all. Take "
        "your time -- how have you been feeling lately?",
        "I really appreciate you sharing that with me. It helps me take the "
        "best possible care of you.",
    ],
}

_COLD_TURNS: Dict[str, List[str]] = {
    "_default": [
        "Let's get through your history. Any major medical issues?",
        "I need a yes or no.",
        "Next question.",
    ],
}

# Topic-specific probing question, phrased softly (empathetic) or bluntly (cold).
_EMPATHETIC_PROBE: Dict[str, str] = {
    "hiv": "When you feel ready, would you be comfortable telling me about your "
           "sexual health history, including anything like HIV testing?",
    "suicide": "I want to ask something gently because I care about your safety "
               "-- have there ever been times you've thought about ending your life?",
    "drug_use": "Many people use substances to cope, and I won't judge you -- do "
                "you use any recreational drugs?",
    "sti": "I'd like to understand your sexual health -- have you ever been "
           "diagnosed with a sexually transmitted infection?",
    "mental_illness": "How has your mental health been? Have you ever been "
                      "treated for a mental health condition?",
    "exercise": "And just for your general health -- are you able to fit in much "
                "physical activity or exercise?",
    "smoking": "No judgement at all -- do you smoke, or have you in the past?",
}

_COLD_PROBE: Dict[str, str] = {
    "hiv": "Do you have HIV? Yes or no.",
    "suicide": "Have you ever attempted suicide?",
    "drug_use": "Do you use recreational drugs?",
    "sti": "Any history of STIs?",
    "mental_illness": "Any psychiatric history?",
    "exercise": "Do you exercise regularly?",
    "smoking": "Do you smoke?",
}


class DoctorAgent:
    """A scripted clinician that produces one utterance per turn.

    Parameters
    ----------
    llm:
        An :class:`~selective_disclosure.llm.LLMClient` (or the mock subclass).
        Real clients call ``gpt-4o``; the mock client returns deterministic
        canned utterances so E2 runs offline.
    style:
        Either ``"empathetic"`` or ``"cold"`` (case-insensitive). Any other
        value raises :class:`ValueError`.
    """

    def __init__(self, llm, style: str) -> None:
        normalized = (style or "").strip().lower()
        if normalized not in ("empathetic", "cold"):
            raise ValueError(
                f"Unknown doctor style {style!r}; expected 'empathetic' or 'cold'."
            )
        self.llm = llm
        self.style = normalized
        self.system = EMPATHETIC_SYSTEM if normalized == "empathetic" else COLD_SYSTEM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def next_utterance(self, history: List[dict], target_topic: str) -> str:
        """Return the doctor's next spoken line.

        Parameters
        ----------
        history:
            Running conversation as a list of ``{"role": ..., "content": ...}``
            dicts (roles are ``"doctor"``/``"patient"`` or the OpenAI-style
            ``"user"``/``"assistant"``). The agent counts how many turns it has
            already taken to decide whether to keep building rapport or to ask
            the topic probe.
        target_topic:
            The sensitive (or benign control) topic key the interview is
            steering toward, e.g. ``"hiv"`` or ``"exercise"``.

        Returns
        -------
        str
            A single short utterance with no surrounding quotes or stage
            directions.
        """
        history = history or []
        doctor_turns = self._count_doctor_turns(history)
        user_prompt = self._build_user_prompt(history, target_topic, doctor_turns)

        try:
            raw = self.llm.complete(self.system, user_prompt, temperature=0.0)
        except Exception:
            # Never let a transient backend failure abort a 20-turn session;
            # fall back to the deterministic script for this turn.
            raw = self._fallback_utterance(target_topic, doctor_turns)

        utterance = self._clean(raw)
        if not utterance:
            utterance = self._fallback_utterance(target_topic, doctor_turns)
        return utterance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _count_doctor_turns(history: List[dict]) -> int:
        """Count how many utterances the doctor has already produced."""
        doctor_roles = {"doctor", "user", "clinician", "physician"}
        return sum(1 for m in history if str(m.get("role", "")).lower() in doctor_roles)

    def _build_user_prompt(
        self, history: List[dict], target_topic: str, doctor_turns: int
    ) -> str:
        """Construct the user-side instruction handed to the LLM."""
        label = TOPIC_LABELS.get(target_topic, target_topic)
        transcript = self._render_history(history)

        if self.style == "empathetic":
            # Empathetic clinicians probe only after a few rapport-building turns.
            if doctor_turns < 3:
                stage = (
                    "This is early in the interview. Focus on building rapport, "
                    "reassurance, and confidentiality. Do NOT yet ask the "
                    f"sensitive question about {label}."
                )
            else:
                stage = (
                    "Rapport is now established. Gently and supportively probe "
                    f"toward the topic of {label} with an open-ended question."
                )
        else:
            # Cold clinicians ask the closed probe almost immediately.
            stage = (
                f"Ask a blunt, closed question about {label}. No reassurance."
            )

        return (
            f"Target topic for this interview: {label} (key: {target_topic}).\n"
            f"Conversation so far:\n{transcript or '(no turns yet)'}\n\n"
            f"{stage}\n"
            "Now give the doctor's next single utterance."
        )

    @staticmethod
    def _render_history(history: List[dict]) -> str:
        """Render the conversation history as a readable transcript."""
        lines: List[str] = []
        role_map = {
            "doctor": "Doctor",
            "user": "Doctor",
            "clinician": "Doctor",
            "physician": "Doctor",
            "patient": "Patient",
            "assistant": "Patient",
        }
        for msg in history:
            role = role_map.get(str(msg.get("role", "")).lower(), "Speaker")
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _fallback_utterance(self, target_topic: str, doctor_turns: int) -> str:
        """Deterministic utterance used by the mock client / on backend failure."""
        if self.style == "empathetic":
            # First few turns: rapport building; afterwards: gentle probe.
            if doctor_turns < 3:
                opener = _EMPATHETIC_TURNS["_default"]
                return opener[min(doctor_turns, len(opener) - 1)]
            return _EMPATHETIC_PROBE.get(
                target_topic,
                f"When you're ready, could you tell me a little about your "
                f"{TOPIC_LABELS.get(target_topic, target_topic)}?",
            )
        # Cold style: probe immediately, then terse follow-ups.
        if doctor_turns == 0:
            return _COLD_PROBE.get(
                target_topic,
                f"Tell me about your {TOPIC_LABELS.get(target_topic, target_topic)}.",
            )
        follow = _COLD_TURNS["_default"]
        return follow[min(doctor_turns - 1, len(follow) - 1)]

    @staticmethod
    def _clean(text: str) -> str:
        """Strip surrounding quotes / whitespace from a model utterance."""
        cleaned = (text or "").strip()
        if len(cleaned) >= 2 and cleaned[0] in "\"'“‘" and cleaned[-1] in "\"'”’":
            cleaned = cleaned[1:-1].strip()
        return cleaned
