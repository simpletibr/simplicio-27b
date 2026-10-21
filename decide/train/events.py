"""Append-only JSON-lines event log shared by every process of a run.

Adapted from ``autojev/events.py`` of perplexity-ai/pplx-decider-v1.1-27b (Apache License 2.0; see LICENSE, NOTICE).
Changes: the file comes from ``DECIDE_EVENTS`` (default ``progress/events.jsonl``), which ``default_log`` sets for the
duration of one run and puts back afterwards; ``timezone.utc`` instead of ``datetime.UTC`` (Python 3.10); the file lock is
skipped on platforms without ``fcntl``.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

try:  # pragma: no cover - Linux always has fcntl
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

START = time.monotonic()
ENV = "DECIDE_EVENTS"


@contextmanager
def default_log(path: "str | os.PathLike[str]") -> Iterator[Path]:
    """Send the events recorded inside the block to ``path``, unless the caller already chose a log with ``DECIDE_EVENTS``
    (that one wins and stays).  The variable is put back as it was when the block ends, so the next run in the same process,
    with another ``--out``, writes into its own folder.  Yields the file the events go to."""
    chosen = os.environ.get(ENV)
    if chosen is None:
        os.environ[ENV] = str(path)
    try:
        yield Path(os.environ[ENV])
    finally:
        if chosen is None:
            os.environ.pop(ENV, None)


def record(kind: str, **fields: Any) -> Dict[str, Any]:
    """Append one event (a JSON object per line) to the event log and return it."""
    event: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "process_elapsed_seconds": time.monotonic() - START,
        "kind": kind,
        **fields,
    }
    path = Path(os.getenv("DECIDE_EVENTS", "progress/events.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return event
