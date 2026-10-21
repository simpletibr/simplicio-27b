"""Decontamination of the training corpus against the public suite (and the private proxy).

A corpus example is rejected when, after NFKC / case-fold / word normalization (and with the JSON keys sorted, so the
order of the keys in either file never matters),

1. the whole request, or one question (state + instructions + options), has the same hash as a reference one;
2. a *line* of at least ``min_line_words`` words equals a reference line;
3. a *short* line (fewer words, at least one letter) of the state or of an instruction equals a short line of the state
   or of an instruction of a reference request: a test item can be a single utterance ("thanks so much!");
4. any window of ``ngram`` (13) words inside a line also occurs in a reference line.

Units (lines and windows) that occur in more than ``max_df`` reference requests are boilerplate (task templates,
shared option lists) and are ignored.  A state is counted once however many questions share it, so a passage with
30 reference questions is not mistaken for a template.  The corpus text is also scanned for the BIG-bench canary,
whatever line breaks it is split by.

Memory: the reference units live in sorted ``array('Q')`` shards (8 bytes per unit), so 100M units fit in ~800 MB.
Everything streams, and a corpus directory is processed part by part so an interrupted run resumes.  The corpus is
read through its ``manifest.json`` (only the parts it lists, sha256 checked); the output directory holds exactly the
parts of the last finished run.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import schema as S
from . import text as T

DEFAULT_NGRAM = 13
DEFAULT_MAX_DF = 20
DEFAULT_MIN_LINE_WORDS = 5
MAX_SKIPPED_FRACTION = 0.01     # a reference file may have this share of rows we cannot read before we stop
#: Bumped when the matching rules change, so parts screened by older rules are not reused by a resumed run.
RULES_VERSION = 2
_SHARDS = 256
_SHIFT = 56


class ReferenceDataError(ValueError):
    """A reference file yields no request, or too many of its rows cannot be read: a silent hole in the screening."""


@dataclass(frozen=True)
class Match:
    reason: str          # exact_request | exact_question | line | short_line | ngram | canary | schema
    detail: str = ""


class _UnitSet:
    """Distinct-request counted set of 64-bit units; after ``freeze`` only units with df <= max_df remain."""

    def __init__(self, max_df: int) -> None:
        self.max_df = max_df
        self._raw: List[array] = [array("Q") for _ in range(_SHARDS)]
        self._kept: Optional[List[array]] = None
        self.total = 0
        self.kept = 0

    def add_request_units(self, units: Iterable[int]) -> None:
        for u in set(units):
            self._raw[u >> _SHIFT].append(u)

    def freeze(self) -> None:
        kept: List[array] = []
        for shard in self._raw:
            values = sorted(shard)
            out = array("Q")
            i = 0
            while i < len(values):
                j = i
                while j < len(values) and values[j] == values[i]:
                    j += 1
                self.total += 1
                if j - i <= self.max_df:
                    out.append(values[i])
                i = j
            self.kept += len(out)
            kept.append(out)
        self._kept = kept
        self._raw = []

    def __contains__(self, unit: int) -> bool:
        assert self._kept is not None
        shard = self._kept[unit >> _SHIFT]
        i = bisect.bisect_left(shard, unit)
        return i < len(shard) and shard[i] == unit


def canonical(request: Mapping[str, Any]) -> Dict[str, Any]:
    """The request with its keys sorted at every level, the form the corpus stores (``schema.canonical_json``)."""
    return json.loads(S.canonical_json(request))


class ReferenceIndex:
    """Hashes of the reference requests; ``check`` says whether a corpus request leaks one of them."""

    def __init__(self, n: int = DEFAULT_NGRAM, max_df: int = DEFAULT_MAX_DF,
                 min_line_words: int = DEFAULT_MIN_LINE_WORDS) -> None:
        if n < 2 or max_df < 1 or min_line_words < 1:
            raise ValueError("n >= 2, max_df >= 1 and min_line_words >= 1 are required")
        self.n, self.max_df, self.min_line_words = n, max_df, min_line_words
        self._requests: Dict[str, str] = {}
        self._questions: Dict[str, str] = {}
        self._lines = _UnitSet(max_df)
        self._grams = _UnitSet(max_df)
        self._shorts = _UnitSet(max_df)
        self._states: set = set()
        self._count = 0
        self._frozen = False
        self.sources: List[Dict[str, Any]] = []

    # ---- units
    def _line_units(self, lines: Sequence[str], statements: bool, out: Tuple[set, set, set]) -> None:
        """Add the units of ``lines`` to ``out`` = (lines, windows, short lines); ``statements``: lines of the state or
        of an instruction, the only ones whose short forms count."""
        lines_out, grams_out, shorts_out = out
        for line in lines:
            toks = T.words(line)
            if len(toks) >= self.min_line_words:
                lines_out.add(T.hash64(" ".join(toks), "line"))
            elif statements and T.has_letters(toks):
                shorts_out.add(T.hash64(" ".join(toks), "short"))
            grams_out.update(T.ngram_hashes(toks, self.n))

    def _keys(self, request: Mapping[str, Any]) -> Tuple[str, Dict[str, str]]:
        texts = T.question_texts(request)
        return S.digest("\n\x1f".join(texts.values())), {qid: S.digest(t) for qid, t in texts.items()}

    # ---- building
    def add(self, request: Mapping[str, Any], ref_id: Optional[str] = None) -> None:
        if self._frozen:
            raise RuntimeError("the index is frozen")
        request = canonical(request)
        rid = ref_id if ref_id is not None else str(self._count)
        whole, per_question = self._keys(request)
        self._requests.setdefault(whole, rid)
        for qid, key in per_question.items():
            self._questions.setdefault(key, f"{rid}#{qid}")
        # a state shared by many questions (a passage with 30 reference questions) counts once for the document
        # frequency: its units are added by the first request that has it
        state_id = int(S.digest(S.canonical_json(request.get("state")))[:16], 16)
        first = state_id not in self._states
        self._states.add(state_id)
        out: Tuple[set, set, set] = (set(), set(), set())
        if first:
            self._line_units(T.state_lines(request), True, out)
        self._line_units(T.instruction_lines(request), True, out)
        self._line_units(T.option_lines(request), False, out)
        self._lines.add_request_units(out[0])
        self._grams.add_request_units(out[1])
        self._shorts.add_request_units(out[2])
        self._count += 1

    def freeze(self) -> None:
        if not self._frozen:
            self._lines.freeze()
            self._grams.freeze()
            self._shorts.freeze()
            self._frozen = True

    # ---- querying
    def check(self, request: Mapping[str, Any]) -> Optional[Match]:
        if not self._frozen:
            raise RuntimeError("freeze() the index before check()")
        request = canonical(request)
        whole, per_question = self._keys(request)
        if whole in self._requests:
            return Match("exact_request", self._requests[whole])
        for key in per_question.values():
            if key in self._questions:
                return Match("exact_question", self._questions[key])
        statements = set(T.state_lines(request)) | set(T.instruction_lines(request))
        lines = [(line, T.words(line)) for line in T.request_lines(request)]
        for line, toks in lines:
            if len(toks) >= self.min_line_words and T.hash64(" ".join(toks), "line") in self._lines:
                return Match("line", line[:120])
        for line, toks in lines:
            if (line in statements and len(toks) < self.min_line_words and T.has_letters(toks)
                    and T.hash64(" ".join(toks), "short") in self._shorts):
                return Match("short_line", line[:120])
        for line, toks in lines:
            hits = sum(1 for g in T.ngram_hashes(toks, self.n) if g in self._grams)
            if hits:
                return Match("ngram", f"{hits} windows of {self.n} words in: {line[:80]}")
        return None

    def stats(self) -> Dict[str, Any]:
        units = {"lines": (self._lines.total, self._lines.kept), "ngrams": (self._grams.total, self._grams.kept),
                 "short_lines": (self._shorts.total, self._shorts.kept)}
        total = sum(t for t, _ in units.values())
        kept = sum(k for _, k in units.values())
        return {"n": self.n, "max_df": self.max_df, "min_line_words": self.min_line_words, "requests": self._count,
                "states": len(self._states), "frozen": self._frozen, "units_total": total, "units_kept": kept,
                "units_dropped_df": total - kept, "rules_version": RULES_VERSION,
                "units": {name: {"total": t, "kept": k} for name, (t, k) in units.items()}}

    def fingerprint(self) -> str:
        """Identifies the index (rules + parameters + reference files) so a resumed run knows its parts are still valid."""
        return S.digest(S.canonical_json({"rules": RULES_VERSION, "n": self.n, "max_df": self.max_df, "min_line_words": self.min_line_words,
                                          "sources": [s["sha256"] for s in self.sources], "requests": self._count}))


# --------------------------------------------------------------------------- readers

def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def _reference_request(row: Any) -> Optional[Dict[str, Any]]:
    """The request of a reference row: suite rows (``state``/``questions`` at the top) or ``{"request": {...}}``
    (an object or its JSON text, as the corpus stores it); ``None`` when the row has neither."""
    if not isinstance(row, dict):
        return None
    if isinstance(row.get("questions"), dict):
        return {"state": row.get("state", {}), "questions": row["questions"]}
    wrapped = row.get("request")
    if isinstance(wrapped, str):
        try:
            wrapped = json.loads(wrapped)
        except ValueError:
            return None
    if isinstance(wrapped, dict) and isinstance(wrapped.get("questions"), dict):
        return {"state": wrapped.get("state", {}), "questions": wrapped["questions"]}
    return None


def iter_reference_requests(path: os.PathLike, stats: Optional[Dict[str, int]] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """(id, request) of every row of a reference file or directory.

    Accepted: decision-index suite rows (``*.jsonl[.gz]`` with state/questions at the top, or a ``request`` object),
    corpus parquet files, and directories holding either.  ``stats`` (``rows``, ``skipped``) counts the JSONL rows
    read and the ones that hold no request, so the caller can refuse a reference it only half understood.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file() and (f.suffix == ".parquet" or ".jsonl" in f.name))
        for f in files:
            yield from iter_reference_requests(f, stats)
        return
    if p.suffix == ".parquet":
        for row in S.read_corpus(p):
            if stats is not None:
                stats["rows"] = stats.get("rows", 0) + 1
            yield row["example_id"], S.parse_row(row)[0]
        return
    with _open_text(p) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            request = _reference_request(row)
            if stats is not None:
                stats["rows"] = stats.get("rows", 0) + 1
                stats["skipped"] = stats.get("skipped", 0) + (request is None)
            if request is not None:
                yield str(row.get("id", f"{p.name}:{i}")), request


def build_index(paths: Sequence[os.PathLike], *, n: int = DEFAULT_NGRAM, max_df: int = DEFAULT_MAX_DF,
                min_line_words: int = DEFAULT_MIN_LINE_WORDS) -> ReferenceIndex:
    index = ReferenceIndex(n=n, max_df=max_df, min_line_words=min_line_words)
    for path in paths:
        before = index._count
        stats: Dict[str, int] = {"rows": 0, "skipped": 0}
        for rid, request in iter_reference_requests(path, stats):
            index.add(request, rid)
        added = index._count - before
        if added == 0 or stats["skipped"] > MAX_SKIPPED_FRACTION * stats["rows"]:
            raise ReferenceDataError(
                f"reference {path}: {added} requests indexed, {stats['skipped']} of {stats['rows']} rows hold no request "
                "(expected suite rows with state/questions, rows with a `request`, or corpus parquet); "
                "screening against a reference that was not read would let every leak through")
        files = [Path(path)] if Path(path).is_file() else sorted(f for f in Path(path).rglob("*") if f.is_file())
        index.sources.append({"path": str(path), "requests": added, "rows": stats["rows"], "skipped": stats["skipped"],
                              "sha256": S.digest("".join(S.sha256_file(f) for f in files))})
    index.freeze()
    return index


# --------------------------------------------------------------------------- corpus

def _screen(row: Mapping[str, Any], index: ReferenceIndex) -> Optional[Match]:
    problems = S.validate_corpus_row(row)
    if problems:
        return Match("schema", problems[0])
    request, _ = S.parse_row(row)
    if T.request_has_canary(request):
        return Match("canary", "BIG-bench canary text")
    return index.check(request)


def decontaminate_corpus(corpus_in: os.PathLike, out_dir: os.PathLike, index: ReferenceIndex, *,
                         resume: bool = True) -> Dict[str, Any]:
    """Write the clean rows of ``corpus_in`` to ``out_dir`` part by part; returns (and saves) the report."""
    out = Path(out_dir)
    if out.resolve() == Path(corpus_in).resolve():
        raise ValueError("--out must be a different directory from --corpus: the clean corpus never overwrites its input")
    # the input is read through its manifest.json (only the parts it lists, sha256 checked), before anything in out changes
    files = S.corpus_files(corpus_in, require_manifest=True)
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        raise ValueError(f"the parts of {corpus_in} must have distinct file names (found {sorted(n for n in set(names) if names.count(n) > 1)})")
    out.mkdir(parents=True, exist_ok=True)
    for stale in (S.MANIFEST_NAME, "decontamination.json"):      # describes the previous run; rewritten at the end
        (out / stale).unlink(missing_ok=True)
    fingerprint = index.fingerprint()
    parts_report: List[Dict[str, Any]] = []
    reasons: Dict[str, int] = {}
    by_source: Dict[str, Dict[str, int]] = {}
    resumed = 0
    for file in files:
        sha = S.sha256_file(file)
        marker = out / (file.name + ".done.json")
        if resume and marker.exists():
            done = json.loads(marker.read_text(encoding="utf-8"))
            output_ok = done.get("rows_out") == 0 or ((out / file.name).exists() and S.sha256_file(out / file.name) == done.get("output_sha256"))
            if done.get("input_sha256") == sha and done.get("fingerprint") == fingerprint and output_ok:
                parts_report.append(done)
                resumed += 1
                _accumulate(done, reasons, by_source)
                continue
        kept: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        src_counts: Dict[str, Dict[str, int]] = {}
        rows_in = 0
        for row in S.read_corpus(file):
            rows_in += 1
            hit = _screen(row, index)
            counts = src_counts.setdefault(row["source"], {"in": 0, "kept": 0, "rejected": 0})
            counts["in"] += 1
            if hit is None:
                kept.append(row)
                counts["kept"] += 1
            else:
                counts["rejected"] += 1
                rejected.append({"example_id": row["example_id"], "source": row["source"], "reason": hit.reason, "detail": hit.detail})
        info = S.write_corpus_part(out / file.name, kept) if kept else {"name": file.name, "rows": 0, "sha256": None}
        if not kept:
            (out / file.name).unlink(missing_ok=True)
        rej_path = out / (file.name + ".rejected.jsonl")
        tmp = rej_path.with_name(rej_path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
        os.replace(tmp, rej_path)
        done = {"part": file.name, "input_sha256": sha, "fingerprint": fingerprint, "rows_in": rows_in,
                "rows_out": len(kept), "rejected": len(rejected), "output_sha256": info["sha256"],
                "reasons": _count(r["reason"] for r in rejected), "by_source": src_counts}
        S.write_manifest(marker, done)
        parts_report.append(done)
        _accumulate(done, reasons, by_source)
    removed = _prune_outputs(out, names, [p["part"] for p in parts_report if p["rows_out"]])
    report = {
        "schema_version": S.SCHEMA_VERSION, "index": index.stats(), "references": index.sources, "removed_stale": removed,
        "parts": len(parts_report), "parts_resumed": resumed,
        "rows_in": sum(p["rows_in"] for p in parts_report), "rows_out": sum(p["rows_out"] for p in parts_report),
        "rejected": sum(p["rejected"] for p in parts_report), "reasons": dict(sorted(reasons.items())),
        "by_source": dict(sorted(by_source.items())),
    }
    S.write_manifest(out / "decontamination.json", report)
    manifest_parts = [{"name": p["part"], "rows": p["rows_out"], "sha256": p["output_sha256"]} for p in parts_report if p["rows_out"]]
    # `sources` so that proxy_private can still check, on this manifest, that the proxy shares no source with the corpus
    sources = {key: {"rows": counts["kept"]} for key, counts in sorted(by_source.items()) if counts["kept"]}
    S.write_manifest(out / "manifest.json", {"schema_version": S.SCHEMA_VERSION, "kind": "corpus", "rows": report["rows_out"],
                                             "parts": manifest_parts, "sources": sources, "decontamination": "decontamination.json"})
    return report


def _prune_outputs(out: Path, input_parts: Sequence[str], kept_parts: Sequence[str]) -> List[str]:
    """Delete what earlier runs left in ``out`` for parts that are no longer in the input (a rebuilt corpus with fewer
    shards, a source that left it): clean parts, markers, rejection lists and half-written files."""
    inputs, kept = set(input_parts), set(kept_parts)
    removed: List[str] = []
    for f in sorted(out.iterdir()):
        name = f.name
        if not f.is_file():
            continue
        base = name[: -len(".done.json")] if name.endswith(".done.json") else name[: -len(".rejected.jsonl")] if name.endswith(".rejected.jsonl") else None
        stale = ((name.endswith(".parquet") and name not in kept)
                 or (base is not None and base not in inputs)
                 or name.endswith((".parquet.tmp", ".json.tmp", ".jsonl.tmp")))
        if stale:
            f.unlink()
            removed.append(name)
    return removed


def _count(items: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def _accumulate(done: Mapping[str, Any], reasons: Dict[str, int], by_source: Dict[str, Dict[str, int]]) -> None:
    for reason, n in done.get("reasons", {}).items():
        reasons[reason] = reasons.get(reason, 0) + n
    for source, counts in done.get("by_source", {}).items():
        agg = by_source.setdefault(source, {"in": 0, "kept": 0, "rejected": 0})
        for k in agg:
            agg[k] += counts.get(k, 0)


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Remove from the corpus everything that overlaps the public suite or the proxy.")
    ap.add_argument("--corpus", required=True, help="corpus parquet file or directory (build_corpus output)")
    ap.add_argument("--out", required=True, help="output directory for the clean corpus, rejected ids and the report")
    ap.add_argument("--reference", action="append", required=True,
                    help="suite rows (*.jsonl.gz), a suite directory, or a corpus/proxy parquet; repeatable")
    ap.add_argument("--ngram", type=int, default=DEFAULT_NGRAM)
    ap.add_argument("--max-df", type=int, default=DEFAULT_MAX_DF)
    ap.add_argument("--min-line-words", type=int, default=DEFAULT_MIN_LINE_WORDS)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    index = build_index(args.reference, n=args.ngram, max_df=args.max_df, min_line_words=args.min_line_words)
    report = decontaminate_corpus(args.corpus, args.out, index, resume=not args.no_resume)
    print(json.dumps({k: report[k] for k in ("rows_in", "rows_out", "rejected", "reasons", "parts", "parts_resumed")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
