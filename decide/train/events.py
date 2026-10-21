"""Append-only JSON-lines event log shared by every process of a run.

Adapted from ``autojev/events.py`` of perplexity-ai/pplx-decider-v1.1-27b (Apache License 2.0; see LICENSE, NOTICE).
Changes: the file comes from ``DECIDE_EVENTS`` (default ``progress/events.jsonl``); ``timezone.utc`` instead of
``datetime.UTC`` (Python 3.10); the file lock is skipped on platforms without ``fcntl``.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - Linux always has fcntl
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

START = time.monotonic()


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
