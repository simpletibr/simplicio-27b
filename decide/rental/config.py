"""Loader of decide/config.rental.yaml (the ``rental:`` namespace; it never touches the kit's keys)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config.rental.yaml"


def load_rental_config(path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - message only
        raise ImportError("PyYAML is required to read config.rental.yaml (install the pinned packages of decide/train/requirements.txt)") from exc
    data = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8")) or {}
    if set(data) != {"rental"}:
        raise ValueError(f"config.rental.yaml must have exactly one top-level key, 'rental' (found {sorted(data)})")
    return data
