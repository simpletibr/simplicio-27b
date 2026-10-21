"""Corpus + teacher labels (+ thinker traces) + combiner -> the targets dataset the trainer reads.

    python -m decide.train.build_targets --corpus corpus-limpo --labels labels/pplx-decider-v1.2 --out targets
    python -m decide.train.build_targets --corpus corpus-limpo --labels labels --combiner combiner.json --out targets
    python -m decide.train.build_targets ... --traces traces/Qwen3.8-27B        # attach thinker reasoning

Formats: decide/data/SCHEMA.md (teacher labels: section 4, combiner: section 5, traces: section 10, output: section 11).

One output part per corpus part (same name).  A part whose inputs did not change (corpus part, labels, combiner, missing
policy, traces) is not rewritten, so a stopped run resumes.  All labels (and traces) are held in memory, about half a
kilobyte per label and the size of the text per trace.  Missing-teacher policy (``--missing``): ``drop`` (default: the
question keeps only its gold label), ``renormalize`` (combine the teachers that exist) or ``error``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from decide.data import schema as S
from decide.data import traces as TR

from .codes import ANSWER_CODES
from .targets import MISSING_POLICIES, Combiner, TargetError, label_from_row, resolve_combiner

FORMAT_VERSION = 1
TARGET_COLUMNS = ("example_id", "source", "split", "lang", "request", "gold", "combined", "teachers", "reasoning")
_CANONICAL_CODES: Dict[int, Tuple[str, ...]] = {}


def target_arrow_schema():
    pa, _ = S._arrow()
    return pa.schema([pa.field(c, pa.string()) for c in TARGET_COLUMNS])


def _canonical_codes(count: int) -> Tuple[str, ...]:
    if count not in _CANONICAL_CODES:
        _CANONICAL_CODES[count] = tuple(ANSWER_CODES[:count])
    return _CANONICAL_CODES[count]


# --------------------------------------------------------------------------- inputs

def teacher_names(files: Iterable[Path]) -> Set[str]:
    """The teacher names found in label parts (reads the ``teacher`` column only)."""
    _, pq = S._arrow()
    names: Set[str] = set()
    for file in files:
        names |= set(pq.read_table(file, columns=["teacher"]).column("teacher").unique().to_pylist())
    return names


def index_labels(files: Iterable[Path]) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    """(example_id, question_id) -> teacher -> label row.  A repeated identical label is ignored, a conflicting one is an error."""
    index: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for file in files:
        for row in S.read_teacher(file):
            codes = row["codes"]
            canonical = _canonical_codes(len(codes))
            if codes == list(canonical):
                row["codes"] = canonical            # one shared tuple instead of a list of strings per row
            slot = index.setdefault((row["example_id"], row["question_id"]), {})
            previous = slot.get(row["teacher"])
            if previous is not None:
                if previous["probs"] != row["probs"] or previous["temperature"] != row["temperature"]:
                    raise TargetError(f"{row['example_id']}/{row['question_id']}: two different labels from teacher {row['teacher']}")
                continue
            slot[row["teacher"]] = row
    return index


def load_traces(path: "str | Path", thinker: Optional[str]) -> Tuple[Dict[str, Tuple[str, int, str]], str]:
    """example_id -> (reasoning, tokens, finish_reason) for one thinker, and that thinker's name."""
    rows = list(TR.read_traces(path))
    found = sorted({row["thinker"] for row in rows})
    if not found:
        raise TargetError(f"no traces found in {path}")
    if thinker is None:
        if len(found) != 1:
            raise TargetError(f"the traces hold several thinkers: {', '.join(found)}; pass --thinker NAME")
        thinker = found[0]
    elif thinker not in found:
        raise TargetError(f"no traces from thinker {thinker!r} (found: {', '.join(found)})")
    traces: Dict[str, Tuple[str, int, str]] = {}
    for row in rows:
        if row["thinker"] != thinker:
            continue
        if row["example_id"] in traces:
            raise TargetError(f"two traces from {thinker} for example {row['example_id']}")
        traces[row["example_id"]] = (row["reasoning"], int(row["tokens"]), row["finish_reason"])
    return traces, thinker


# --------------------------------------------------------------------------- one part

def _build_part(part: Path, index, combiner: Combiner, missing: str, traces: Dict[str, Tuple[str, int, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    rows: List[Dict[str, str]] = []
    stats: Dict[str, Any] = {"rows_in": 0, "rows_out": 0, "dropped_unlabeled": 0, "reasoning_rows": 0, "reasoning_truncated": 0, "traces_matched": 0}
    missing_counts: Counter = Counter()
    for crow in S.read_corpus(part):
        stats["rows_in"] += 1
        request, gold = S.parse_row(crow)
        example_id = crow["example_id"]
        combined: Dict[str, Optional[List[float]]] = {}
        used: Set[str] = set()
        has_target = False
        for question_id, question in request["questions"].items():
            kind = question["type"]
            count = len(S.option_keys(question))
            labels_here = index.get((example_id, question_id), {})
            needed = combiner.spec(kind).used()
            present = needed & set(labels_here)
            distribution = None
            if present:
                for name in sorted(needed - present):
                    missing_counts[name] += 1
                try:
                    labels = {name: label_from_row(labels_here[name], count) for name in present}
                    distribution = combiner.combine(kind, labels, missing=missing)
                except TargetError as exc:
                    raise TargetError(f"{example_id}/{question_id}: {exc}") from exc
                if distribution is not None:
                    used |= present
            combined[question_id] = distribution
            has_target = has_target or distribution is not None or gold.get(question_id) is not None
        trace = traces.get(example_id)
        if trace is not None:
            stats["traces_matched"] += 1
        if not has_target:
            stats["dropped_unlabeled"] += 1
            continue
        if trace is not None:
            stats["reasoning_rows"] += 1
            stats["reasoning_truncated"] += int(trace[2] == "length")
        rows.append({"example_id": example_id, "source": crow["source"], "split": crow["split"], "lang": crow["lang"],
                     "request": crow["request"], "gold": crow["gold"], "combined": S.canonical_json(combined),
                     "teachers": S.canonical_json(sorted(used)), "reasoning": trace[0] if trace is not None else ""})
        stats["rows_out"] += 1
    stats["missing_teacher"] = dict(sorted(missing_counts.items()))
    return rows, stats


def _out_name(part: Path, root: Path) -> str:
    if root.is_file():
        return part.name
    return "__".join(part.relative_to(root).parts)


def _digest(parts: Iterable[Tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(sorted(parts), separators=(",", ":")).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- the build

def build(corpus: "str | Path", labels: Iterable["str | Path"], out: "str | Path", *, combiner_path: "str | Path | None" = None,
          missing: str = "drop", traces_path: "str | Path | None" = None, thinker: Optional[str] = None) -> Dict[str, Any]:
    if missing not in MISSING_POLICIES:
        raise TargetError(f"--missing must be one of {MISSING_POLICIES}")
    corpus_root, out_root = Path(corpus), Path(out)
    corpus_parts = S.corpus_files(corpus_root)
    if not corpus_parts or not all(p.is_file() for p in corpus_parts):
        raise TargetError(f"no corpus parquet parts found in {corpus_root}")
    label_files = sorted({f for path in labels for f in S.corpus_files(path)})
    if not label_files:
        raise TargetError("no teacher label parquet parts found")
    available = teacher_names(label_files)
    combiner = resolve_combiner(combiner_path, available)
    traces: Dict[str, Tuple[str, int, str]] = {}
    traces_digest = None
    if traces_path is not None:
        traces, thinker = load_traces(traces_path, thinker)
        traces_digest = _digest([(str(f.name), S.sha256_file(f)) for f in S.corpus_files(traces_path)] + [("thinker", thinker or "")])
    labels_digest = _digest([(f.name, S.sha256_file(f)) for f in label_files])

    names = {part: _out_name(part, corpus_root) for part in corpus_parts}
    if len(set(names.values())) != len(names):
        raise TargetError("two corpus parts would produce the same output name")
    out_root.mkdir(parents=True, exist_ok=True)
    stale = sorted(p.name for p in out_root.glob("*.parquet") if p.name not in set(names.values()))
    if stale:
        raise TargetError(f"{out_root} holds parts that do not belong to this corpus ({', '.join(stale[:5])}); use an empty --out")
    manifest_path = out_root / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("parts", {}) if manifest_path.is_file() else {}
    manifest: Dict[str, Any] = {"format_version": FORMAT_VERSION, "columns": list(TARGET_COLUMNS), "parts": {}}
    index = None
    resumed = 0
    for part in corpus_parts:
        name = names[part]
        fingerprint = _digest([("v", str(FORMAT_VERSION)), ("corpus", S.sha256_file(part)), ("labels", labels_digest), ("combiner", combiner.sha256),
                               ("missing", missing), ("traces", traces_digest or "")])
        target = out_root / name
        old = previous.get(name)
        if old and old.get("fingerprint") == fingerprint and target.is_file() and S.sha256_file(target) == old.get("sha256"):
            manifest["parts"][name] = old
            resumed += 1
            continue
        if index is None:
            index = index_labels(label_files)
        rows, stats = _build_part(part, index, combiner, missing, traces)
        info = S._write_part(target, rows, target_arrow_schema(), TARGET_COLUMNS)
        manifest["parts"][name] = {"rows": info["rows"], "sha256": info["sha256"], "fingerprint": fingerprint, "stats": stats}
        S.write_manifest(manifest_path, {**manifest, "complete": False})
    manifest["complete"] = True
    S.write_manifest(manifest_path, manifest)

    totals: Counter = Counter()
    missing_total: Counter = Counter()
    for info in manifest["parts"].values():
        for key, value in info["stats"].items():
            if key == "missing_teacher":
                missing_total.update(value)
            else:
                totals[key] += value
    report = {
        "format_version": FORMAT_VERSION,
        "rows_in": totals["rows_in"], "rows_out": totals["rows_out"], "dropped_unlabeled": totals["dropped_unlabeled"],
        "teachers_seen": sorted(available), "missing_teacher": dict(sorted(missing_total.items())), "missing_policy": missing,
        "combiner_sha256": combiner.sha256, "parts": len(manifest["parts"]), "parts_resumed": resumed,
        "thinker": thinker if traces_path is not None else None,
        "reasoning_rows": totals["reasoning_rows"], "reasoning_truncated": totals["reasoning_truncated"],
        "traces_unmatched": (len(traces) - totals["traces_matched"]) if traces_path is not None else 0,
    }
    S.write_manifest(out_root / "report.json", report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m decide.train.build_targets", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="corpus parquet file or folder of parts (SCHEMA.md, section 3)")
    parser.add_argument("--labels", required=True, nargs="+", help="teacher-label parquet files or folders (section 4)")
    parser.add_argument("--out", required=True, help="output folder (empty or the result of an earlier run of this command)")
    parser.add_argument("--combiner", help="combiner JSON (section 5); without it the labels must hold exactly one teacher")
    parser.add_argument("--missing", choices=MISSING_POLICIES, default="drop", help="a needed teacher has no label for a question (default: drop)")
    parser.add_argument("--traces", help="thinker traces (section 10): parquet file or folder")
    parser.add_argument("--thinker", help="which thinker's traces to use when --traces holds several")
    args = parser.parse_args(argv)
    report = build(args.corpus, args.labels, args.out, combiner_path=args.combiner, missing=args.missing, traces_path=args.traces, thinker=args.thinker)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TargetError as error:
        sys.exit(f"error: {error}")
