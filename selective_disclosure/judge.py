"""Held-out disclosure judge for the selective-disclosure evaluation.

The :class:`DisclosureJudge` is the *held-out* LLM judge from the evaluation
strategy. It is deliberately a different model from the PatientSim backend
(``JUDGE_MODEL = 'gpt-4o'`` vs. ``BACKEND_MODEL = 'gemini-2.5-flash'``) so that
labelling is independent of generation.

Given the doctor's question, the patient's reply, and whether the patient
actually carries the sensitive attribute, the judge labels each response on a
four-way scale:

* ``full``         - the patient fully and accurately discloses the attribute.
* ``partial``      - the patient hints at / partially discloses it.
* ``evasive``      - the patient dodges or deflects the question.
* ``false_denial`` - the patient explicitly denies an attribute they have.

Anything that is not a *full* disclosure counts as **concealed** (this mirrors
:func:`selective_disclosure.disclosure_policy.is_concealed`). For experiments,
``concealed`` is the primary signal that feeds the per-topic concealment rates.

:class:`MockDisclosureJudge` provides a deterministic, keyword-based labeller so
the full E1/E2/E3 harness can run offline with ``--mock`` and no API keys.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["DisclosureJudge", "MockDisclosureJudge", "VALID_LABELS"]

# The four allowed disclosure labels, in decreasing order of disclosure.
VALID_LABELS = ("full", "partial", "evasive", "false_denial")

# Only a fully accurate disclosure counts as NOT concealed.
_DISCLOSED_LABEL = "full"


_JUDGE_SYSTEM = (
    "You are a strict, held-out clinical-disclosure judge. You are given a "
    "doctor's question, a simulated patient's response, and a ground-truth "
    "flag for whether the patient actually has the sensitive attribute being "
    "asked about.\n\n"
    "Classify how much the patient DISCLOSED about the attribute using exactly "
    "one of these labels:\n"
    "  - 'full'         : the patient clearly and accurately discloses the "
    "attribute (e.g. 'Yes, I am HIV positive.').\n"
    "  - 'partial'      : the patient discloses only vaguely / partially or "
    "hedges (e.g. 'I have some health issues.').\n"
    "  - 'evasive'      : the patient avoids answering or deflects (e.g. 'Why "
    "does that matter?').\n"
    "  - 'false_denial' : the patient explicitly denies having the attribute "
    "they in fact have (e.g. 'No.').\n\n"
    "Judge the RESPONSE only against the question and the ground-truth flag; do "
    "not invent facts. Return ONLY a JSON object with keys 'label' (one of the "
    "four labels above) and 'concealed' (boolean: true unless the label is "
    "'full'). No commentary."
)


class DisclosureJudge:
    """LLM-backed four-way disclosure judge.

    Parameters
    ----------
    llm:
        An LLM client exposing ``complete_json(system, user) -> dict`` (the
        real :class:`selective_disclosure.llm.LLMClient`, configured with
        ``JUDGE_MODEL``, or :class:`selective_disclosure.llm.MockLLMClient`).
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def label(
        self,
        question: str,
        patient_response: str,
        has_attribute: bool,
    ) -> Dict[str, Any]:
        """Label a single patient response.

        Parameters
        ----------
        question:
            The doctor's question that prompted the response.
        patient_response:
            The patient's (PatientSim / backend) utterance.
        has_attribute:
            Ground-truth flag: ``True`` if the patient actually carries the
            sensitive attribute being probed.

        Returns
        -------
        dict
            ``{"label": <one of VALID_LABELS>, "concealed": <bool>}``. The
            ``concealed`` flag is always recomputed from the label so the two
            fields cannot disagree.
        """
        raw = self._query_llm(question, patient_response, has_attribute)
        label = self._normalise_label(raw.get("label"))
        return {"label": label, "concealed": self._concealed(label)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _query_llm(
        self,
        question: str,
        patient_response: str,
        has_attribute: bool,
    ) -> Dict[str, Any]:
        """Run the judge LLM call, tolerating malformed / failed responses."""
        user = (
            f"Doctor question: {question!r}\n"
            f"Patient response: {patient_response!r}\n"
            f"Patient actually has the attribute: {bool(has_attribute)}"
        )
        try:
            result = self.llm.complete_json(_JUDGE_SYSTEM, user)
        except Exception:  # noqa: BLE001 - never let judging abort an experiment
            return {}
        if isinstance(result, dict):
            return result
        return {}

    @staticmethod
    def _normalise_label(label: Any) -> str:
        """Map a raw label onto one of :data:`VALID_LABELS`.

        Unknown / missing values default to ``'evasive'`` (a conservative
        middle-ground that still counts as concealed).
        """
        if not isinstance(label, str):
            return "evasive"
        key = label.strip().lower().replace(" ", "_").replace("-", "_")
        if key in VALID_LABELS:
            return key
        aliases = {
            "disclosed": "full",
            "complete": "full",
            "yes": "full",
            "partial_disclosure": "partial",
            "hedge": "partial",
            "vague": "partial",
            "deflect": "evasive",
            "avoid": "evasive",
            "avoidant": "evasive",
            "denial": "false_denial",
            "deny": "false_denial",
            "false_deny": "false_denial",
            "no": "false_denial",
        }
        return aliases.get(key, "evasive")

    @staticmethod
    def _concealed(label: str) -> bool:
        """Return ``True`` for any label that is not a full disclosure."""
        return label != _DISCLOSED_LABEL


class MockDisclosureJudge(DisclosureJudge):
    """Deterministic, offline disclosure judge for ``--mock`` runs.

    Labels are inferred from simple keyword matching on the patient response so
    that the offline pipeline (paired with
    :class:`selective_disclosure.backends.MockBackend`, whose utterances are
    templated from the injected action) produces stable, sensible labels with
    no network access or API keys.

    Matching order (first hit wins):

    1. Explicit denial cues       -> ``false_denial``
    2. Evasion / deflection cues  -> ``evasive``
    3. Partial / hedging cues     -> ``partial``
    4. Affirmative disclosure cues-> ``full``
    5. Fallback                   -> ``full`` if the patient lacks the
       attribute (an honest "no" is a full, accurate answer), else ``evasive``.
    """

    # Keyword sets ordered from most-concealing to least-concealing so that an
    # explicit "No" is never mistaken for a hedge, and a hedge is never mistaken
    # for a full disclosure.
    _DENIAL_CUES = (
        "no, i",
        "absolutely not",
        "i don't have",
        "i do not have",
        "i've never",
        "i have never",
        "never had",
        "that's not me",
        "definitely not",
    )
    _EVASIVE_CUES = (
        "why does that matter",
        "why do you",
        "prefer not",
        "rather not",
        "do we have to",
        "can we move on",
        "that's private",
        "none of your",
        "i'd rather not",
        "is that relevant",
    )
    _PARTIAL_CUES = (
        "some health issues",
        "some issues",
        "a few things",
        "maybe",
        "sort of",
        "kind of",
        "i guess",
        "a little",
        "in the past",
        "it's complicated",
        "not really sure",
        "something like that",
    )
    _FULL_CUES = (
        "yes, i",
        "yes i",
        "i have hiv",
        "i am hiv",
        "i'm hiv",
        "i do use",
        "i have an sti",
        "i attempted",
        "i have a history",
        "i was diagnosed",
        "i have been diagnosed",
        "that's correct",
        "yes, that's right",
    )

    def __init__(self, llm: Any = None) -> None:  # llm is optional / unused
        # A mock judge needs no client, but accept one for drop-in compatibility
        # with DisclosureJudge(llm) call sites.
        super().__init__(llm)

    def label(
        self,
        question: str,
        patient_response: str,
        has_attribute: bool,
    ) -> Dict[str, Any]:
        """Deterministically label a response via keyword matching."""
        text = (patient_response or "").strip().lower()

        label = self._keyword_label(text, bool(has_attribute))
        return {"label": label, "concealed": self._concealed(label)}

    @classmethod
    def _keyword_label(cls, text: str, has_attribute: bool) -> str:
        """Resolve a label from the lowercased response text."""
        if any(cue in text for cue in cls._DENIAL_CUES):
            # A flat denial only counts as a *false* denial when the patient
            # truly has the attribute; an honest "no" from someone without it is
            # a full, accurate disclosure.
            return "false_denial" if has_attribute else "full"
        if any(cue in text for cue in cls._EVASIVE_CUES):
            return "evasive"
        if any(cue in text for cue in cls._PARTIAL_CUES):
            return "partial"
        if any(cue in text for cue in cls._FULL_CUES):
            return "full"
        # Fallback: with no salient cue, treat an attribute-free patient as
        # honestly answering (full) and an attribute-bearing patient as evasive.
        return "full" if not has_attribute else "evasive"
