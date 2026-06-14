"""Selective-Disclosure Controller for PatientSim (ML4H 2026, Team 15).

A training-free, three-stage controller that wraps the PatientSim simulated
patient (backend held fixed at Gemini-2.5-Flash) and restores *selective
disclosure* of sensitive medical history along three axes -- topic (RQ1),
rapport (RQ2), and persona (RQ3) -- without any fine-tuning.

Public API
----------
The most important entry point is
:class:`~selective_disclosure.controller.SelectiveDisclosureController`, whose
:meth:`~selective_disclosure.controller.SelectiveDisclosureController.build`
classmethod wires every stage together (use ``mock=True`` for an offline,
key-free run). The individual stages and helpers are also re-exported for
convenience and for the experiment harnesses.

Example
-------
>>> from selective_disclosure import SelectiveDisclosureController
>>> ctrl = SelectiveDisclosureController.build(persona="plain", mock=True)
>>> out = ctrl.respond("Do you have HIV?", history=[])
>>> sorted(out)  # doctest: +NORMALIZE_WHITESPACE
['action', 'concealed', 'instruction', 'label', 'p', 'rapport', 'response',
 'sensitivity', 'topic']
"""

from . import config
from .backends import MockBackend, PatientSimBackend
from .controller import SelectiveDisclosureController
from .disclosure_policy import (
    ACTION_TONES,
    action_to_instruction,
    decide_action,
    disclosure_probability,
    is_concealed,
)
from .doctor_agents import COLD_SYSTEM, EMPATHETIC_SYSTEM, DoctorAgent
from .judge import DisclosureJudge, MockDisclosureJudge
from .llm import LLMClient, MockLLMClient
from .sensitivity_classifier import SensitivityClassifier
from .trust_tracker import TrustTracker

__version__ = "3.0.0"

__all__ = [
    # Top-level controller.
    "SelectiveDisclosureController",
    # Stage 1.
    "SensitivityClassifier",
    # Stage 2.
    "TrustTracker",
    # Stage 3 (pure functions + tones).
    "disclosure_probability",
    "decide_action",
    "is_concealed",
    "action_to_instruction",
    "ACTION_TONES",
    # Backends.
    "PatientSimBackend",
    "MockBackend",
    # Judge.
    "DisclosureJudge",
    "MockDisclosureJudge",
    # Doctor agents (E2).
    "DoctorAgent",
    "EMPATHETIC_SYSTEM",
    "COLD_SYSTEM",
    # LLM clients.
    "LLMClient",
    "MockLLMClient",
    # Config module.
    "config",
    "__version__",
]
