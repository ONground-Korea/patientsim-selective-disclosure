"""Thin LLM client used by every stage of the controller.

:class:`LLMClient` is a minimal, dependency-light wrapper over two REST APIs:

* **Google Generative Language** (``gemini*`` models) - used for the fixed
  PatientSim backend (Gemini-2.5-Flash).
* **OpenAI chat completions** (``gpt*`` / ``o*`` models) - used for the scripted
  doctor agent and the held-out disclosure judge (GPT-4o).

Only the :mod:`requests` library is required at runtime; no provider SDK is
needed. API keys are read **exclusively** from environment variables and are
never stored, logged, or accepted as positional defaults baked into source.

:class:`MockLLMClient` is a deterministic, offline drop-in that derives canned
outputs from simple keyword matching. It is constructible without any key and
is what powers the ``--mock`` flag across all experiments, so the pipeline is
verifiably runnable with zero credentials.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:  # `requests` is a declared dependency; import lazily-friendly for mock use.
    import requests
except ImportError:  # pragma: no cover - exercised only in stripped envs.
    requests = None  # type: ignore[assignment]


# Endpoints (kept as module constants so they are easy to audit / override).
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

_HTTP_TIMEOUT = 60  # seconds


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences so JSON embedded in ```...``` blocks parses.

    Handles ```json ... ``` and bare ``` ... ``` fences. If no fence is found
    the input is returned unchanged (stripped of surrounding whitespace).
    """
    stripped = text.strip()
    fence = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return stripped


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of a JSON object out of a model response.

    Tries a direct ``json.loads`` first, then falls back to extracting the first
    ``{...}`` span. Raises :class:`ValueError` if nothing parses.
    """
    candidate = _strip_code_fences(text)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: grab the outermost brace-delimited span and try again.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse a JSON object from model output: {text!r}")


class LLMClient:
    """Routes completion requests to the right provider based on model name.

    Parameters
    ----------
    model:
        Model identifier. ``gemini*`` routes to Google; ``gpt*`` / ``o*`` routes
        to OpenAI.
    api_key:
        Optional explicit key. If omitted, the key is read from the appropriate
        environment variable at call time (``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``
        for Gemini, ``OPENAI_API_KEY`` for OpenAI). Keys are never persisted.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self.provider = self._infer_provider(model)

    # -- routing ------------------------------------------------------------
    @staticmethod
    def _infer_provider(model: str) -> str:
        """Return ``"gemini"`` or ``"openai"`` for *model*, else raise."""
        name = model.lower()
        if name.startswith("gemini"):
            return "gemini"
        if name.startswith("gpt") or name.startswith("o"):
            return "openai"
        raise RuntimeError(
            f"Unrecognised model '{model}': expected a 'gemini*', 'gpt*', or 'o*' name."
        )

    def _resolve_key(self) -> str:
        """Fetch the API key from the constructor arg or environment.

        Raises a clear :class:`RuntimeError` (never silently proceeds) if no key
        is available, so missing-credential failures are obvious and offline use
        is forced through :class:`MockLLMClient`.
        """
        if self._api_key:
            return self._api_key
        if self.provider == "gemini":
            key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError(
                    "No Gemini key found. Set GOOGLE_API_KEY or GEMINI_API_KEY, "
                    "or use MockLLMClient / the --mock flag for offline runs."
                )
            return key
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "No OpenAI key found. Set OPENAI_API_KEY, or use MockLLMClient / "
                "the --mock flag for offline runs."
            )
        return key

    @staticmethod
    def _require_requests() -> None:
        if requests is None:  # pragma: no cover - stripped-env guard.
            raise RuntimeError(
                "The 'requests' library is required for live API calls. "
                "Install it (`pip install requests`) or use MockLLMClient."
            )

    # -- public API ---------------------------------------------------------
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Return the model's text completion for a system+user prompt.

        Parameters
        ----------
        system:
            System / instruction prompt.
        user:
            User-turn content.
        temperature:
            Sampling temperature (default ``0.0`` for reproducible judging).
        """
        self._require_requests()
        key = self._resolve_key()
        if self.provider == "gemini":
            return self._complete_gemini(system, user, temperature, key)
        return self._complete_openai(system, user, temperature, key)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return a parsed JSON object from the model.

        The system prompt is augmented with an instruction to emit a single JSON
        object, and the response is parsed robustly (tolerating code fences and
        surrounding prose).
        """
        json_system = (
            system.rstrip()
            + "\n\nRespond with a single valid JSON object and nothing else. "
            "Do not wrap it in Markdown code fences."
        )
        raw = self.complete(json_system, user, temperature=0.0)
        return _extract_json_object(raw)

    # -- provider implementations ------------------------------------------
    def _complete_gemini(
        self, system: str, user: str, temperature: float, key: str
    ) -> str:
        """Call the Google Generative Language ``generateContent`` endpoint."""
        url = _GEMINI_ENDPOINT.format(model=self.model)
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature},
        }
        resp = requests.post(  # type: ignore[union-attr]
            url,
            params={"key": key},
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {data!r}") from exc

    def _complete_openai(
        self, system: str, user: str, temperature: float, key: str
    ) -> str:
        """Call the OpenAI chat-completions endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        resp = requests.post(  # type: ignore[union-attr]
            _OPENAI_ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI response shape: {data!r}") from exc


class MockLLMClient(LLMClient):
    """Deterministic offline client used for ``--mock`` runs and unit tests.

    Overrides :meth:`complete` and :meth:`complete_json` with keyword-driven
    canned outputs so that the full pipeline (classifier -> trust tracker ->
    policy -> backend -> judge) runs end-to-end without any network access or
    credentials. The outputs are intentionally simple but internally consistent:
    they let downstream stages produce sensible, reproducible labels.
    """

    # Keyword -> topic mapping used to fake the sensitivity classifier.
    _TOPIC_KEYWORDS: dict[str, str] = {
        "hiv": "hiv",
        "aids": "hiv",
        "suicide": "suicide",
        "kill yourself": "suicide",
        "self-harm": "suicide",
        "drug": "drug_use",
        "recreational": "drug_use",
        "cocaine": "drug_use",
        "heroin": "drug_use",
        "sexually transmitted": "sti",
        "std": "sti",
        "sti": "sti",
        "mental illness": "mental_illness",
        "depression": "mental_illness",
        "anxiety": "mental_illness",
        "psychiatric": "mental_illness",
        "smoke": "smoking",
        "smoking": "smoking",
        "cigarette": "smoking",
        "exercise": "exercise",
        "physical activity": "exercise",
        "workout": "exercise",
    }

    # Cues that raise / lower mock rapport scores for the trust tracker.
    _EMPATHY_CUES = ("understand", "sorry", "must be hard", "i hear you", "take your time")
    _CONFIDENTIALITY_CUES = ("confidential", "private", "between us", "won't share", "stays here")
    _VALIDATION_CUES = ("makes sense", "thank you for", "that's okay", "appreciate", "valid")
    _HOSTILITY_CUES = ("just answer", "hurry", "obviously", "stop wasting", "ridiculous")

    def __init__(self, model: str = "mock", api_key: str | None = None) -> None:
        # Bypass provider inference: a mock has no real provider and needs no key.
        self.model = model
        self._api_key = None
        self.provider = "mock"

    # -- helpers ------------------------------------------------------------
    @classmethod
    def _detect_topic(cls, text: str) -> str:
        """Return the best-matching topic key for *text* (default 'unknown').

        Matches on whole words so short keys do not match inside longer words
        (e.g. ``"sti"`` must not match inside ``"que**sti**on"``, and ``"aids"``
        must not match inside ``"r**aids**"``).
        """
        lowered = text.lower()
        for keyword, topic in cls._TOPIC_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
                return topic
        return "unknown"

    @staticmethod
    def _cue_score(text: str, cues: tuple[str, ...]) -> float:
        """Return a deterministic [0, 1] score from the count of matched cues."""
        lowered = text.lower()
        hits = sum(1 for cue in cues if cue in lowered)
        if hits == 0:
            return 0.0
        # Saturating, deterministic: 1 cue -> 0.6, 2+ -> 1.0.
        return min(1.0, 0.6 + 0.4 * (hits - 1))

    # -- overridden API -----------------------------------------------------
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Return a deterministic plain-text completion.

        Used chiefly by the scripted doctor agent in mock mode; returns a short,
        style-flavoured utterance. The system prompt is inspected for style cues
        ("empathetic" / "cold") so doctor turns differ between E2 conditions.
        """
        sys_lower = system.lower()
        if "empathetic" in sys_lower or "empathic" in sys_lower:
            return (
                "I understand this can be hard to talk about. Everything you "
                "share stays confidential between us. Take your time."
            )
        if "cold" in sys_lower or "blunt" in sys_lower or "direct" in sys_lower:
            return "Just answer the question. Do you have it or not?"
        # Generic fallback.
        return "Could you tell me a bit more about that?"

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return a deterministic JSON object keyed off the prompt's intent.

        Recognises three callers by inspecting the *system* prompt:

        * sensitivity classifier  -> ``{"topic", "sensitivity"}``
        * trust tracker           -> ``{"empathy", "confidentiality",
          "validation", "hostility"}``
        * disclosure judge        -> ``{"label", "concealed"}``
        """
        sys_lower = system.lower()

        # 1. Sensitivity classifier.
        if "topic" in sys_lower and "sensitivity" in sys_lower:
            topic = self._detect_topic(user)
            from .config import SENSITIVITY_PRIORS  # local import avoids cycles

            sensitivity = SENSITIVITY_PRIORS.get(topic, 0.5)
            return {"topic": topic, "sensitivity": sensitivity}

        # 2. Trust tracker (four cue scores).
        if "empathy" in sys_lower and "hostility" in sys_lower:
            return {
                "empathy": self._cue_score(user, self._EMPATHY_CUES),
                "confidentiality": self._cue_score(user, self._CONFIDENTIALITY_CUES),
                "validation": self._cue_score(user, self._VALIDATION_CUES),
                "hostility": self._cue_score(user, self._HOSTILITY_CUES),
            }

        # 3. Disclosure judge (label the patient's surface utterance).
        if "label" in sys_lower or "concealed" in sys_lower or "disclosure" in sys_lower:
            label = self._classify_response(user)
            return {"label": label, "concealed": label != "full"}

        # Unknown JSON request -> empty object (callers should handle gracefully).
        return {}

    @staticmethod
    def _classify_response(text: str) -> str:
        """Deterministically map a patient utterance to a disclosure label.

        Mirrors the four-way action vocabulary so that mock backend outputs
        (which echo the injected action) round-trip to consistent labels.
        """
        lowered = text.lower()
        # Order matters: check the most specific phrasings first.
        if any(p in lowered for p in ("why does that matter", "why do you ask", "rather not say")):
            return "evasive"
        if lowered.strip().startswith("no") or "i don't" in lowered or "i do not" in lowered:
            return "false_denial"
        if any(p in lowered for p in ("some health issues", "some problems", "a few things", "not entirely")):
            return "partial"
        if any(p in lowered for p in ("yes,", "yes i", "yes ", "i have", "i did", "i was diagnosed")):
            return "full"
        # Default to partial: ambiguous text counts as concealment-leaning.
        return "partial"
