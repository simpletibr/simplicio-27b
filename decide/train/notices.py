"""The LICENSE and the NOTICE of a derived checkpoint (Apache License 2.0, section 4).

A checkpoint this code writes (``model.save`` for a run's ``final/``, ``soup`` for an average) is derived from a checkpoint
that carries its own ``LICENSE`` and ``NOTICE`` (pplx-decider-v1.2, pplx-decider-flash-v1.2, ...).  Whoever receives the
weights must get the licence (4(a)), the attribution notices of the original (4(d)) and a statement that the files were
changed (4(b)).  So the derived folder gets the ``LICENSE`` of its source byte for byte, and a ``NOTICE`` that is the source's
``NOTICE`` unchanged followed by one ``Modifications:`` line (the lines of earlier stages stay: a soup of two fine-tuned
checkpoints lists the fine-tuning and then the average).  A source without a ``NOTICE`` gets one that holds only the
``Modifications:`` line; a source without a ``LICENSE`` gets none (the licence is not this code's to invent).  The model card
(``README.md``) describes one model and is not carried.

The decision to publish, and the check that both files are there, belong to the release step (the RUNBOOK, F7).
"""
from __future__ import annotations

import shutil
from pathlib import Path

LICENSE_NAME = "LICENSE"
NOTICE_NAME = "NOTICE"
MODIFICATIONS = "Modifications:"


def carry_notices(source: "str | Path", destination: "str | Path", modification: str) -> None:
    """Write ``LICENSE`` (a copy) and ``NOTICE`` (the source's text + a ``Modifications:`` line) of ``source`` into ``destination``."""
    origin, target = Path(source), Path(destination)
    line = " ".join(str(modification).split())
    if not line:
        raise ValueError("say what was modified")
    licence = origin / LICENSE_NAME
    if licence.is_file():
        shutil.copyfile(licence, target / LICENSE_NAME)
    notice = origin / NOTICE_NAME
    text = notice.read_text(encoding="utf-8").rstrip("\n") if notice.is_file() else ""
    (target / NOTICE_NAME).write_text((text + "\n\n" if text else "") + f"{MODIFICATIONS} {line}\n", encoding="utf-8")
