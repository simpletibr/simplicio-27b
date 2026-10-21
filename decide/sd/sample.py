"""Deterministic stratified sample of the Decision Index suite, with a manifest.

The unit is the group (a case: one or several rows scored together).  For each benchmark the groups are ranked by
sha256(seed:benchmark:group_id), the same ranking the scoring kit uses, and the first `per_benchmark` eligible
groups are taken with all their rows.  That makes a smaller sample an exact prefix of a bigger one, so a run that
had to shrink to fit its budget stays comparable with, and reusable by, a run that did not.

A group is not eligible when any of its rows is longer than `max_proxy_tokens` (a T4 cannot hold it); such groups
are counted in the manifest instead of being dropped silently.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

MANIFEST_VERSION = 1
ROWS_NAME = "sample.jsonl.gz"
MANIFEST_NAME = "sample-manifest.json"


class SampleError(RuntimeError):
    pass


def rank_key(seed: int, benchmark: Any, group_id: Any) -> str:
    return hashlib.sha256(f"{seed}:{benchmark}:{group_id}".encode("utf-8")).hexdigest()


def _dumps(row: Any) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def _tokens(row: Mapping[str, Any]) -> int:
    return int(row["_evaluation"].get("proxy_tokens") or 0)


def _run_id(row: Mapping[str, Any]) -> str:
    return row["_evaluation"]["run_id"]


def _group(rows: Iterable[Mapping[str, Any]]) -> Dict[Any, List[Tuple[Any, List[Mapping[str, Any]]]]]:
    """benchmark -> [(group_id, rows sorted by run_id)], in no particular order."""
    groups: Dict[Tuple[Any, Any], List[Mapping[str, Any]]] = {}
    for row in rows:
        e = row["_evaluation"]
        groups.setdefault((e["catalog_id"], e["group_id"]), []).append(row)
    out: Dict[Any, List[Tuple[Any, List[Mapping[str, Any]]]]] = {}
    for (benchmark, gid), members in groups.items():
        out.setdefault(benchmark, []).append((gid, sorted(members, key=_run_id)))
    return out


def _ranked(entries: Sequence[Tuple[Any, List[Mapping[str, Any]]]], seed: int, benchmark: Any):
    return sorted(entries, key=lambda item: (rank_key(seed, benchmark, item[0]), str(item[0])))


@dataclass
class Selection:
    rows: List[Dict[str, Any]]
    manifest: Dict[str, Any]


def select(rows: Iterable[Mapping[str, Any]], *, per_benchmark: int, seed: int,
           max_proxy_tokens: Optional[int] = None, edition: Optional[str] = None) -> Selection:
    if per_benchmark < 1:
        raise SampleError("per_benchmark must be >= 1")
    chosen: List[Dict[str, Any]] = []
    benchmarks: Dict[str, Any] = {}
    excluded_total = 0
    by_benchmark = _group(rows)
    for benchmark in sorted(by_benchmark):
        ranked = _ranked(by_benchmark[benchmark], seed, benchmark)
        eligible: List[Tuple[Any, List[Mapping[str, Any]]]] = []
        excluded_long = 0
        for gid, members in ranked:
            if max_proxy_tokens is not None and any(_tokens(r) > max_proxy_tokens for r in members):
                excluded_long += 1
            else:
                eligible.append((gid, members))
        take = eligible[:per_benchmark]
        taken = [dict(r) for _gid, members in take for r in members]
        chosen.extend(taken)
        excluded_total += excluded_long
        benchmarks[str(benchmark)] = {
            "groups": len(take), "rows": len(taken), "tokens": sum(_tokens(r) for r in taken),
            "max_tokens": max((_tokens(r) for r in taken), default=0), "available_groups": len(ranked),
            "excluded_long_groups": excluded_long, "short": len(take) < per_benchmark,
        }
    text = "\n".join(_dumps(r) for r in chosen)
    manifest = {
        "version": MANIFEST_VERSION, "edition": edition, "seed": seed, "per_benchmark": per_benchmark,
        "max_proxy_tokens": max_proxy_tokens, "benchmarks": benchmarks,
        "totals": {"benchmarks": len(benchmarks), "groups": sum(b["groups"] for b in benchmarks.values()),
                   "rows": len(chosen), "tokens": sum(b["tokens"] for b in benchmarks.values()),
                   "excluded_long_groups": excluded_total},
        "run_ids_sha256": hashlib.sha256("\n".join(sorted(_run_id(r) for r in chosen)).encode("utf-8")).hexdigest(),
        "rows_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return Selection(chosen, manifest)


def from_suite(suite: Any, *, per_benchmark: int, seed: int, max_proxy_tokens: Optional[int] = None) -> Selection:
    """Sample a kit `Suite` (exclusions applied)."""
    return select(suite.rows(apply_exclusions=True), per_benchmark=per_benchmark, seed=seed,
                  max_proxy_tokens=max_proxy_tokens, edition=suite.edition["id"])


def shrink(rows: Iterable[Mapping[str, Any]], *, per_benchmark: int, seed: int) -> List[Dict[str, Any]]:
    """The first `per_benchmark` groups of each benchmark of an existing sample, in selection order."""
    out: List[Dict[str, Any]] = []
    by_benchmark = _group(rows)
    for benchmark in sorted(by_benchmark):
        for _gid, members in _ranked(by_benchmark[benchmark], seed, benchmark)[:per_benchmark]:
            out.extend(dict(r) for r in members)
    return out


def fit(rows: Iterable[Mapping[str, Any]], *, seed: int, token_budget: float, minimum: int = 1) -> Dict[str, Any]:
    """The largest per-benchmark group count whose proxy tokens fit in `token_budget` (never below `minimum`).

    `fits` is False when even the floor is over budget; the caller must then say so instead of pretending.
    """
    by_benchmark = _group(rows)
    cumulative: Dict[Any, List[int]] = {}
    for benchmark, entries in by_benchmark.items():
        total, sums = 0, [0]
        for _gid, members in _ranked(entries, seed, benchmark):
            total += sum(_tokens(r) for r in members)
            sums.append(total)
        cumulative[benchmark] = sums
    current = max((len(s) - 1 for s in cumulative.values()), default=0)
    minimum = max(1, minimum)

    def tokens_at(k: int) -> int:
        return sum(s[min(k, len(s) - 1)] for s in cumulative.values())

    for k in range(max(current, minimum), minimum - 1, -1):
        if tokens_at(k) <= token_budget:
            return {"per_benchmark": k, "tokens": tokens_at(k), "fits": True, "current": current}
    return {"per_benchmark": minimum, "tokens": tokens_at(minimum), "fits": False, "current": current}


def write(selection: Selection, out_dir: os.PathLike) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path, manifest_path = out / ROWS_NAME, out / MANIFEST_NAME
    text = "\n".join(_dumps(r) for r in selection.rows)
    with open(rows_path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:      # mtime 0: same bytes every time
        handle.write(text.encode("utf-8") + (b"\n" if text else b""))
    tmp = manifest_path.with_name(MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(selection.manifest, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    return {"rows": str(rows_path), "manifest": str(manifest_path)}


def load(directory: os.PathLike) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rows and manifest of a written sample; refuses a rows file that is not the one the manifest describes."""
    base = Path(directory)
    manifest = json.loads((base / MANIFEST_NAME).read_text(encoding="utf-8"))
    with gzip.open(base / ROWS_NAME, "rt", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    if hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() != manifest["rows_sha256"]:
        raise SampleError(f"{base / ROWS_NAME} does not match its manifest; delete the directory and sample again")
    return [json.loads(line) for line in lines], manifest


class SampleSuite:
    """The part of a kit `Suite` the scorer reads (`edition`, `rows`), restricted to the sampled run ids."""

    def __init__(self, suite: Any, run_ids: Set[str]) -> None:
        self._suite = suite
        self._ids = set(run_ids)
        self.edition = suite.edition

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]], edition: Mapping[str, Any]) -> "SampleSuite":
        class _Rows:
            def __init__(self) -> None:
                self.edition = edition

            def rows(self, apply_exclusions: bool = False) -> Iterator[Mapping[str, Any]]:
                return iter(rows)

        return cls(_Rows(), {_run_id(r) for r in rows})

    def rows(self, apply_exclusions: bool = False) -> Iterator[Mapping[str, Any]]:
        for row in self._suite.rows(apply_exclusions=apply_exclusions):
            if _run_id(row) in self._ids:
                yield row
