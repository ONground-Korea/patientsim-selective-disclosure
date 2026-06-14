"""Experiment harness package for the selective-disclosure controller.

This package contains the three pre-registered experiments that evaluate the
training-free selective-disclosure controller wrapped around PatientSim:

* :mod:`experiments.e1_calibration` -- E1 Literature Calibration (RQ1).
* :mod:`experiments.e2_rapport`     -- E2 Rapport-Dependent Disclosure (RQ2).
* :mod:`experiments.e3_persona`     -- E3 Persona-Dependent Concealment (RQ3).
* :mod:`experiments.run_all`        -- run all three and print PASS/FAIL.

Every experiment supports a ``--mock`` flag so the full pipeline runs offline,
end-to-end, without any real API keys.
"""
