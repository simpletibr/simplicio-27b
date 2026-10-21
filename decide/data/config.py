"""Loader of decide/config.data.yaml (the ``data:`` namespace; it never touches the kit's keys)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config.data.yaml"


def load_data_config(path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - message only
        raise ImportError("PyYAML is required to read config.data.yaml: pip install pyyaml") from exc
    data = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8")) or {}
    if set(data) != {"data"}:
        raise ValueError(f"config.data.yaml must have exactly one top-level key, 'data' (found {sorted(data)})")
    return data
