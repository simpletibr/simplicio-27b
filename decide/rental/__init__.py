"""The rented GPU day (F6): profiles, estimator, the day's orchestrator, gate G4 and the optional thinker-trace step.

Everything here was written and tested on a CPU with no GPU, no Hugging Face account and no Modal account: the logic is
verified, the behaviour on a rented machine is **UNVERIFIED** until the owner runs it (see decide/RUNBOOK.md).

    python -m decide.rental.estimate --profile econ_b200            # hours and dollars, every number UNVERIFIED
    python -m decide.rental.run --profile econ_b200 --workdir DIR   # the plan of the day (add --execute to run it)
"""
