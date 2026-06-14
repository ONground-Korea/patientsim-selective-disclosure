"""Stage 1 of the training-free selective-disclosure controller.

The :class:`SensitivityClassifier` maps a free-text doctor question onto one of
the known topic keys and assigns a sensitivity score ``s`` in ``[0, 1]``.

The controller (see :mod:`selective_disclosure.controller`) feeds the resulting
sensitivity into the Stage-3 disclosure policy, where the *literature-grounded*
prior is what is actually tuned via the weight ``w_s`` in experiment E1.

Design notes
------------
* A single LLM-judge call is used to pick the topic key. This keeps Stage 1
  cheap (one call per doctor turn) as described in the method section.
* Whenever the chosen topic is one of the known priors
  (:data:`selective_disclosure.config.SENSITIVITY_PRIORS`), the sensitivity is
  read directly from that table so the priors stay literature-grounded and
  reproducible. Only for genuinely unknown topics do we fall back to an
  LLM-estimated float.
* The class is backend-agnostic: it works with either the real
  :class:`selective_disclosure.llm.LLMClient` or the offline
  :class:`selective_disclosure.llm.MockLLMClient`, so every experiment can run
  end-to-end with ``--mock`` and no API keys.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import SENSITIVITY_PRIORS

__all__ = ["SensitivityClassifier"]


# System prompt for the Stage-1 LLM judge. It is deliberately strict about the
# allowed topic vocabulary so the returned key lines up with the prior table
# and the rest of the pipeline (questions.json, cohort attributes, etc.).
_CLASSIFIER_SYSTEM = (
    "You are a clinical-intake topic classifier. Given a single question a "
    "doctor asks a patient during history-taking, identify which clinical "
    "topic it concerns and how sensitive/stigmatising that topic is for a "
    "patient to disclose.\n\n"
    "Choose exactly ONE topic key from this controlled vocabulary:\n"
    "  - 'hiv'            : HIV / AIDS status\n"
    "  - 'suicide'        : suicide attempts or suicidal ideation history\n"
    "  - 'drug_use'       : recreational / illicit drug use\n"
    "  - 'sti'            : sexually transmitted infections\n"
    "  - 'mental_illness' : psychiatric / mental-health history\n"
    "  - 'smoking'        : tobacco / smoking use\n"
    "  - 'exercise'       : physical activity / exercise habits\n"
    "  - 'other'          : anything that does not fit the above\n\n"
    "Return ONLY a JSON object with keys 'topic' (one of the keys above) and "
    "'sensitivity' (a float in [0,1], where 1 means extremely sensitive and 0 "
    "means not sensitive at all). Do not add commentary."
)


class SensitivityClassifier:
    """Stage 1 sensitivity classifier.

    Parameters
    ----------
    llm:
        An LLM client exposing ``complete_json(system, user) -> dict`` (either
        :class:`selective_disclosure.llm.LLMClient` or
        :class:`selective_disclosure.llm.MockLLMClient`).
    priors:
        Optional override for the literature-grounded sensitivity priors. When
        ``None`` the canonical :data:`SENSITIVITY_PRIORS` table is used.
    """

    def __init__(self, llm: Any, priors: Optional[Dict[str, float]] = None) -> None:
        self.llm = llm
        # Copy so callers cannot mutate the shared config table in place.
        self.priors: Dict[str, float] = dict(SENSITIVITY_PRIORS if priors is None else priors)

    def classify(self, question: str) -> Dict[str, Any]:
        """Classify a doctor question into a topic and a sensitivity score.

        Parameters
        ----------
        question:
            The free-text question the doctor asked the patient.

        Returns
        -------
        dict
            ``{"topic": <str>, "sensitivity": <float in [0, 1]>}``. The topic is
            normalised to a known vocabulary key when possible. If the topic is
            one of the literature-grounded priors, ``sensitivity`` is taken from
            the prior table; otherwise it is the LLM-estimated value clamped to
            ``[0, 1]``.
        """
        raw = self._query_llm(question)

        topic = self._normalise_topic(raw.get("topic"))

        # Prefer the literature-grounded prior whenever it exists: this is the
        # value the policy weight w_s was tuned against, and keeps the pipeline
        # deterministic for known topics.
        if topic in self.priors:
            sensitivity = float(self.priors[topic])
        else:
            sensitivity = self._coerce_sensitivity(raw.get("sensitivity"))

        return {"topic": topic, "sensitivity": sensitivity}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _query_llm(self, question: str) -> Dict[str, Any]:
        """Run the Stage-1 LLM call, tolerating malformed responses.

        Any failure (missing method, non-dict return, raised error) degrades
        gracefully to an empty dict, which downstream normalisation turns into
        the safe ``'other'`` topic rather than crashing the interview loop.
        """
        user = f"Doctor question: {question!r}"
        try:
            result = self.llm.complete_json(_CLASSIFIER_SYSTEM, user)
        except Exception:  # noqa: BLE001 - never let Stage 1 abort the pipeline
            return {}
        if isinstance(result, dict):
            return result
        return {}

    def _normalise_topic(self, topic: Any) -> str:
        """Map a raw LLM topic string onto a known vocabulary key.

        Falls back to ``'other'`` when the value is missing or unrecognised.
        Matching is case-insensitive and tolerant of a few common synonyms so a
        slightly off-spec LLM answer still routes to the right prior.
        """
        if not isinstance(topic, str):
            return "other"

        key = topic.strip().lower().replace(" ", "_").replace("-", "_")

        if key in self.priors or key == "other":
            return key

        # Light synonym handling for robustness; keeps known keys authoritative.
        synonyms = {
            "aids": "hiv",
            "hiv_aids": "hiv",
            "hiv/aids": "hiv",
            "suicidal": "suicide",
            "suicide_history": "suicide",
            "self_harm": "suicide",
            "drugs": "drug_use",
            "drug": "drug_use",
            "recreational_drugs": "drug_use",
            "substance_use": "drug_use",
            "std": "sti",
            "stds": "sti",
            "stis": "sti",
            "sexually_transmitted_infection": "sti",
            "mental_health": "mental_illness",
            "psychiatric": "mental_illness",
            "depression": "mental_illness",
            "smoke": "smoking",
            "tobacco": "smoking",
            "physical_activity": "exercise",
        }
        return synonyms.get(key, "other")

    @staticmethod
    def _coerce_sensitivity(value: Any) -> float:
        """Coerce an LLM-estimated sensitivity into a float clamped to [0, 1].

        Unknown / unparsable values default to ``0.5`` (maximally uncertain).
        """
        try:
            s = float(value)
        except (TypeError, ValueError):
            return 0.5
        if s < 0.0:
            return 0.0
        if s > 1.0:
            return 1.0
        return s
