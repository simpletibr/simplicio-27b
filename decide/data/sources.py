"""Registry of data sources and the license policy (deny by default).

``sources.yaml`` lists every dataset the pipeline may read, with the license the dataset card declared, the
evidence for it (URL and date of the check) and a decision: ``allow`` (trains the model), ``review`` (only with
``--include-review``: share-alike, no license metadata, restricted upstream content) or ``deny``.
``role: heldout`` sources never train; they build the private proxy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

DEFAULT_SOURCES_PATH = Path(__file__).resolve().parent / "sources.yaml"
DECISIONS = ("allow", "review", "deny")
ROLES = ("train", "heldout")
KINDS = ("benchmark_train", "tasksource", "other")
#: Licenses that allow commercial use without share-alike.  CC BY (attribution) goes in the model card.
LICENSE_ALLOW = frozenset({"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-3.0", "cc-by-4.0"})
#: Sources that must never train a model, whatever their registry entry says: (pattern, reason).
DENYLIST = (
    (re.compile(r"(^|[/_\-\s])(bbh|big[-_ ]?bench([-_ ]?hard)?)($|[/_\-\s])", re.I), "BIG-bench canary: never train on it"),
    (re.compile(r"gpqa", re.I), "GPQA: the authors ask that its questions not be posted in plain text; evaluation data"),
    (re.compile(r"(^|/)hle($|[-_/])", re.I), "HLE: gated evaluation data"),
    (re.compile(r"(^|/)anli$|facebook/anli|adversarial[-_ ]?nli", re.I), "ANLI is CC BY-NC 4.0"),
    (re.compile(r"ragtruth", re.I), "RAGTruth contexts are non-commercial or may not be redistributed"),
)
_BAD_LICENSE = re.compile(r"(^|[-_ ])n[cd]($|[-_ .\d])|non-?commercial|no-?deriv", re.I)


class PolicyError(ValueError):
    """The registry breaks the license policy."""


@dataclass(frozen=True)
class Source:
    key: str
    hf_id: str
    license: str
    decision: str
    evidence: str
    adapter: str
    config: Optional[str] = None
    split: str = "train"
    revision: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    lang: str = "en"
    kind: str = "benchmark_train"
    role: str = "train"
    cap: Optional[int] = None
    fields_verified: bool = False
    notes: str = ""
    #: Repositories that only ship a loading script: direct file URLs instead of parquet, e.g.
    #: {"data": url, "labels": url, "sha256": {"data": hex, "labels": hex}} (read without running any script).
    files: Dict[str, Any] = field(default_factory=dict)


def denylist_reason(source: Source) -> Optional[str]:
    """Why this source may never train a model (hard denylist, license NC/ND), or None."""
    for pattern, reason in DENYLIST:
        if pattern.search(source.hf_id) or pattern.search(source.key):
            return reason
    if _BAD_LICENSE.search(source.license or ""):
        return f"license {source.license} forbids commercial use or derivatives"
    return None


def check_policy(sources: Mapping[str, Source]) -> None:
    for key, s in sources.items():
        if s.decision not in DECISIONS:
            raise PolicyError(f"{key}: decision must be one of {DECISIONS}")
        if s.role not in ROLES:
            raise PolicyError(f"{key}: role must be one of {ROLES}")
        reason = denylist_reason(s)
        if reason and s.decision != "deny":
            raise PolicyError(f"{key}: {reason} (decision must be 'deny')")
        if s.decision == "allow":
            if (s.license or "").lower() not in LICENSE_ALLOW:
                raise PolicyError(f"{key}: license {s.license!r} is not in the allowlist {sorted(LICENSE_ALLOW)}; use 'review' or 'deny'")
            if not s.evidence.strip():
                raise PolicyError(f"{key}: 'allow' needs evidence (URL and date of the license check)")


def usable(source: Source, *, include_review: bool = False, role: str = "train") -> bool:
    if source.role != role:
        return False
    return source.decision == "allow" or (include_review and source.decision == "review")


def load_sources(path: Optional[Path] = None) -> Dict[str, Source]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - message only
        raise ImportError("PyYAML is required to read sources.yaml (install the pinned packages of decide/data/requirements.txt)") from exc
    data = yaml.safe_load(Path(path or DEFAULT_SOURCES_PATH).read_text(encoding="utf-8")) or {}
    out: Dict[str, Source] = {}
    for key, raw in (data.get("sources") or {}).items():
        unknown = set(raw) - set(Source.__dataclass_fields__) - {"key"}
        if unknown:
            raise PolicyError(f"{key}: unknown fields {sorted(unknown)}")
        out[key] = Source(key=key, **raw)
    return out
