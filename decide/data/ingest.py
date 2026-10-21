"""Turn a source (a registry entry + an opened dataset) into corpus rows: adapter, canary check, dedup, cap."""
from __future__ import annotations

import heapq
from typing import Any, Dict, Iterator, List, Mapping, Optional, Protocol, Tuple

from . import schema as S
from . import text as T
from .adapters import ADAPTERS, PREPARERS, AdapterContext
from .sources import Source

PROBE_ROWS = 200


class Dataset(Protocol):
    """What a reader returns: the HF ``features`` (for class names) and a re-iterable stream of row dicts."""

    features: Mapping[str, Any]

    def rows(self) -> Iterator[Mapping[str, Any]]: ...


class AdapterError(RuntimeError):
    """The adapter converts none of the first rows: the field names in sources.yaml are probably wrong."""


def rank(seed: int, key: str, example_id: str) -> int:
    """Deterministic pseudo-random rank used to cap a source to a uniform sample that does not depend on row order."""
    return int(S.digest(f"{seed}:{key}:{example_id}")[:16], 16)


def convert_source(source: Source, dataset: Dataset, *, seed: int, cap: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rows of one source (dicts with example_id, source, license, lang, request, gold) and the conversion stats."""
    if source.adapter not in ADAPTERS:
        raise AdapterError(f"{source.key}: unknown adapter {source.adapter!r}")
    adapter = ADAPTERS[source.adapter]
    args: Dict[str, Any] = dict(source.args)
    if source.adapter in PREPARERS:
        args.update(PREPARERS[source.adapter](dataset.rows(), args))
    stats: Dict[str, Any] = {"seen": 0, "converted": 0, "skipped": 0, "canary": 0, "duplicates": 0, "invalid": 0}
    seen_requests = set()
    kept: List[Tuple[int, int, Dict[str, Any]]] = []     # heap of (-rank, index, row) when capped
    first_keys: List[str] = []
    for idx, raw in enumerate(dataset.rows()):
        stats["seen"] += 1
        if not first_keys:
            first_keys = sorted(raw)
        out = adapter(raw, args, AdapterContext(source.key, idx, seed, dataset.features))
        if out is None:
            stats["skipped"] += 1
            if stats["converted"] == 0 and stats["seen"] >= PROBE_ROWS:
                raise AdapterError(f"{source.key}: none of the first {PROBE_ROWS} rows converts; the row has the fields {first_keys}")
            continue
        request, gold, native = out
        problems = S.validate_request(request) or S.validate_gold(request, gold)
        if problems:
            stats["invalid"] += 1
            continue
        if T.contains_canary(S.canonical_json(request)):
            stats["canary"] += 1
            continue
        digest = S.request_digest(request)
        if digest in seen_requests:
            stats["duplicates"] += 1
            continue
        seen_requests.add(digest)
        stats["converted"] += 1
        example_id = f"{source.key}:{native if native is not None else idx}"
        row = {"example_id": example_id, "source": source.key, "license": source.license, "lang": source.lang,
               "request": request, "gold": gold}
        r = rank(seed, source.key, example_id)
        if cap is None:
            kept.append((0, idx, row))
        elif len(kept) < cap:
            heapq.heappush(kept, (-r, idx, row))
        elif -kept[0][0] > r:
            heapq.heapreplace(kept, (-r, idx, row))
    if stats["seen"] and not stats["converted"]:
        raise AdapterError(f"{source.key}: no row converts; the row has the fields {first_keys}")
    rows = [row for _, _, row in sorted(kept, key=lambda t: t[1])]
    stats["kept"] = len(rows)
    return rows, stats
