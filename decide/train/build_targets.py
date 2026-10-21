"""Corpus + teacher labels (+ thinker traces) + combiner -> the targets dataset the trainer reads.

    python -m decide.train.build_targets --corpus corpus-limpo --labels labels/pplx-decider-v1.2 --out targets
    python -m decide.train.build_targets --corpus corpus-limpo --labels labels --combiner combiner.json --out targets
    python -m decide.train.build_targets ... --traces traces/Qwen3.8-27B        # attach thinker reasoning

Formats: decide/data/SCHEMA.md (teacher labels: section 4, combiner: section 5, traces: section 10, output: section 11).

One output part per corpus part (same name).  A part whose inputs did not change (corpus part, labels, combiner, missing
policy, traces) is not rewritten, so a stopped run resumes.  All labels (and traces) are held in memory, about half a
kilobyte per label and the size of the text per trace.  Missing-teacher policy (``--missing``): ``drop`` (default: the
question keeps only its gold label), ``renormalize`` (combine the teachers that exist) or ``error``.

Like ``build_corpus``, ``--out`` holds exactly the result of the last finished run: ``manifest.json`` (the parts, with rows and
sha256) is removed when a run starts and written when it ends, the trainer reads only the parts it lists, and parts and
markers a run does not list are deleted (``removed_stale``).

Where the rows come from (SCHEMA.md, section 11).  ``--corpus`` must be the folder ``decontaminate`` left: its ``manifest.json``
says ``tool: "decontaminate"`` and names ``decontamination.json``, which carries the same stamp.  A raw ``build_corpus`` folder, a
folder sealed by hand (``seal_directory``) and a loose parquet file are refused, because a manifest only proves that the list of
parts is whole, not that the rows were screened against the public suites and the private sets.  ``--allow-undecontaminated`` is
the explicit way out; either way ``corpus: {tool, manifest_sha256, decontamination_sha256, allow_undecontaminated}`` goes into the
manifest and the report of the targets, the trainer copies it into the checkpoint, and it is part of what a finished part was built
from.  ``--labels`` and ``--traces`` folders are read through their ``manifest.json`` too (``seal_directory`` makes one), so a part an
earlier build left behind never reaches the targets.  A trace carries the ``request_sha256`` of the request it was made for, and a
trace made for another request is an error.

What a run may overwrite and delete is limited to what it can prove it wrote (SCHEMA.md, section 1): ``--out`` is a folder of
its own that does not overlap any input (``--out`` = the corpus would rewrite its parts as targets, ``--out`` = a label folder
would delete the labels; symlinks and ``..`` are resolved first), the ``manifest.json`` and ``report.json`` it replaces carry its
``tool`` stamp, and the parts it replaces or removes are the ones its own markers and the manifest of its last finished run name,
by plain file name (a manifest that names a path is refused).  Any other file in ``--out`` stays, a file with the name of a part
included: the run refuses instead of overwriting it.
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
from .protocol import has_reasoning_block
from .targets import MISSING_POLICIES, Combiner, TargetError, label_from_row, resolve_combiner

FORMAT_VERSION = 1
TOOL = "build_targets"      # the `tool` stamp of its manifest.json and report.json: how a later run knows it wrote them
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


def load_traces(files: Iterable[Path], thinker: Optional[str]) -> Tuple[Dict[str, Tuple[str, int, str, str]], str]:
    """example_id -> (reasoning, tokens, finish_reason, request_sha256) for one thinker, and that thinker's name.

    ``files`` are the parts the manifest of the traces folder lists (``schema.corpus_files(path, require_manifest=True)``)."""
    rows = [row for file in files for row in TR.read_traces(file)]
    found = sorted({row["thinker"] for row in rows})
    if not found:
        raise TargetError("no traces found")
    if thinker is None:
        if len(found) != 1:
            raise TargetError(f"the traces hold several thinkers: {', '.join(found)}; pass --thinker NAME")
        thinker = found[0]
    elif thinker not in found:
        raise TargetError(f"no traces from thinker {thinker!r} (found: {', '.join(found)})")
    traces: Dict[str, Tuple[str, int, str, str]] = {}
    for row in rows:
        if row["thinker"] != thinker:
            continue
        if row["example_id"] in traces:
            raise TargetError(f"two traces from {thinker} for example {row['example_id']}")
        traces[row["example_id"]] = (row["reasoning"], int(row["tokens"]), row["finish_reason"], row["request_sha256"])
    return traces, thinker


# --------------------------------------------------------------------------- where the rows come from

DECONTAMINATE_TOOL = "decontaminate"          # the `tool` stamp of the manifest and of decontamination.json that decontaminate leaves
DECONTAMINATION_REPORT = "decontamination.json"
LOOSE_CORPUS = {"tool": None, "manifest_sha256": None, "decontamination_sha256": None}


def _json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def corpus_provenance(corpus: "str | Path", *, allow_undecontaminated: bool = False) -> Dict[str, Any]:
    """``{tool, manifest_sha256, decontamination_sha256, allow_undecontaminated}`` of the corpus, or :class:`TargetError`.

    The corpus is clean when its ``manifest.json`` says ``tool: "decontaminate"`` and names ``decontamination.json``, and that file
    exists, is a JSON object and carries the same ``tool`` stamp.  Anything else (a raw ``build_corpus`` folder, a folder sealed by hand
    with ``seal_directory``, a loose parquet file, a stamp without its report) is refused unless ``allow_undecontaminated``, which is
    recorded as passed.  What is known about a corpus that is refused but allowed is recorded as it is (``None`` for what does not exist)."""
    root = Path(corpus)
    allowed = {"allow_undecontaminated": bool(allow_undecontaminated)}
    way_out = "run `python -m decide.data.decontaminate` on it and pass its --out, or pass --allow-undecontaminated (it is recorded in the targets)"
    if not root.is_dir():
        if not allow_undecontaminated:
            raise TargetError(f"--corpus {root} is a parquet file, not the folder decontaminate leaves: nothing says the rows were screened; {way_out}")
        return {**LOOSE_CORPUS, **allowed}
    manifest_path = root / S.MANIFEST_NAME
    manifest = _json_file(manifest_path)
    if manifest is None:
        raise TargetError(f"{manifest_path} is missing or unreadable: the corpus is not the output of a finished build")
    tool = manifest.get("tool")
    report_path = root / DECONTAMINATION_REPORT
    report = _json_file(report_path) if manifest.get("decontamination") == DECONTAMINATION_REPORT else None
    clean = tool == DECONTAMINATE_TOOL and report is not None and report.get("tool") == DECONTAMINATE_TOOL
    if not clean and not allow_undecontaminated:
        if tool != DECONTAMINATE_TOOL:
            raise TargetError(f"{manifest_path} was made by {tool!r}, not by {DECONTAMINATE_TOOL}: the rows were not screened against the public suites and "
                              f"the private sets; {way_out}")
        raise TargetError(f"{manifest_path} says {DECONTAMINATE_TOOL} but {DECONTAMINATION_REPORT} is missing, is not named by the manifest or was not written "
                          f"by {DECONTAMINATE_TOOL}; {way_out}")
    return {"tool": tool if isinstance(tool, str) else None, "manifest_sha256": S.sha256_file(manifest_path),
            "decontamination_sha256": S.sha256_file(report_path) if clean else None, **allowed}


# --------------------------------------------------------------------------- one part

def _build_part(part: Path, index, combiner: Combiner, missing: str, traces: Dict[str, Tuple[str, int, str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    rows: List[Dict[str, str]] = []
    stats: Dict[str, Any] = {"rows_in": 0, "rows_out": 0, "dropped_unlabeled": 0, "reasoning_rows": 0, "reasoning_truncated": 0, "traces_matched": 0,
                             "reasoning_state_has_block": 0, "questions": 0, "unlabeled_questions": 0}
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
            stats["questions"] += 1
            if not present and missing == "error":
                raise TargetError(f"{example_id}/{question_id}: no label from teacher(s) {', '.join(sorted(needed))} (--missing error)")
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
            stats["unlabeled_questions"] += distribution is None
            has_target = has_target or distribution is not None or gold.get(question_id) is not None
        trace = traces.get(example_id)
        if trace is not None:
            stats["traces_matched"] += 1
            if trace[3] != S.request_digest(request):
                raise TargetError(f"{example_id}: the trace was made for another request (its request_sha256 {trace[3][:12]}... is not the "
                                  f"{S.request_digest(request)[:12]}... of the corpus request): a reasoning is not attached to a request it was not written for")
        if not has_target:
            stats["dropped_unlabeled"] += 1
            continue
        if trace is not None and has_reasoning_block(request["state"]):
            stats["reasoning_state_has_block"] += 1       # the state already holds a `Raciocínio:` block: it cannot take a second one
            trace = None
        if trace is not None:
            stats["reasoning_rows"] += 1
            stats["reasoning_truncated"] += int(trace[2] == "length")
        rows.append({"example_id": example_id, "source": crow["source"], "split": crow["split"], "lang": crow["lang"],
                     "request": crow["request"], "gold": crow["gold"], "combined": S.canonical_json(combined),
                     "teachers": S.canonical_json(sorted(used)), "reasoning": trace[0] if trace is not None else ""})
        stats["rows_out"] += 1
    stats["missing_teacher"] = dict(sorted(missing_counts.items()))
    return rows, stats


def _read_marker(path: Path) -> Optional[Dict[str, Any]]:
    """The entry a finished part left (``name``, ``rows``, ``sha256``, ``fingerprint``, ``stats``), or None if there is none or it is unreadable."""
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (isinstance(entry, dict) and {"name", "rows", "sha256", "fingerprint", "stats"} <= set(entry)):
        return None
    return entry if entry["name"] + ".done.json" == path.name else None        # a marker copied under another name is not trusted


def _out_name(part: Path, root: Path) -> str:
    if root.is_file():
        return part.name
    return "__".join(part.relative_to(root).parts)


def _digest(parts: Iterable[Tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(sorted(parts), separators=(",", ":")).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- the build

def _marker(out_root: Path, name: str) -> Path:
    return out_root / (name + ".done.json")


def _check_out(out_root: Path, inputs: Iterable[Path]) -> None:
    """``--out`` is a folder of its own: not a file, not an input, not inside one and not holding one (a single input file counts as itself)."""
    if out_root.exists() and not out_root.is_dir():
        raise TargetError(f"--out {out_root} is a file: the targets need a folder of their own")
    here = out_root.resolve()
    for path in inputs:
        there = path.resolve()
        if there == here or there.is_relative_to(here) or here.is_relative_to(there):
            raise TargetError(f"--out {out_root} overlaps the input {path}: the targets go to a folder of their own and never overwrite or sit inside "
                              "their inputs")


def _assert_report_is_ours(out_root: Path) -> None:
    path = out_root / "report.json"
    if not path.is_file():
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        report = None
    if not isinstance(report, dict) or report.get("tool") != TOOL:
        raise TargetError(f"{path} was not written by {TOOL}: it will not delete it; use another --out")


def _plain_name(name: Any) -> bool:
    """A part of ``--out`` is a file directly in it: a non-empty name with no directory, no ``..`` and no drive."""
    return isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name


def _listed_names(manifest: Optional[Dict[str, Any]]) -> Set[str]:
    """The part names of the manifest of an earlier run; :class:`TargetError` when one is not a plain file name (a forged or hand-edited
    manifest must not make the builder delete a file outside ``--out``)."""
    names: Set[str] = set()
    for entry in (manifest or {}).get("parts", []):
        name = entry.get("name") if isinstance(entry, dict) else None
        if not _plain_name(name):
            raise TargetError(f"the manifest.json in --out lists the part {name!r}, which is not a plain file name: build_targets did not write it and "
                              "will not delete anything because of it; use another --out")
        names.add(name)
    return names


def _owned(out_root: Path, previous: Optional[Dict[str, Any]]) -> Set[str]:
    """The parts in ``out_root`` that this builder wrote: the ones its last finished run lists and the ones a finished part left a marker for."""
    owned = _listed_names(previous)
    for marker in sorted(out_root.glob("*.parquet.done.json")):
        entry = _read_marker(marker)
        if entry is not None:
            owned.add(entry["name"])
    return owned


def _prune(out_root: Path, keep: Set[str], previous: Optional[Dict[str, Any]]) -> List[str]:
    """Delete what earlier runs of this builder left in ``out_root`` that this run does not list: parts, their markers and half-written files.

    Only what it can prove it wrote is touched: the parts its own markers name and the parts the manifest of its last finished run
    (``previous``) lists, with the companions it derives from their names.  Anything else in ``out_root`` stays."""
    owned = _owned(out_root, previous)
    removed: List[str] = []

    def drop(name: str) -> None:
        file = out_root / name
        if file.is_file():
            file.unlink()
            removed.append(name)

    for name in sorted(owned - keep):
        drop(name)
        drop(name + ".done.json")
    leftovers = [S.MANIFEST_NAME, "report.json"]
    for name in sorted(owned | keep):
        leftovers += [name, name + ".done.json"]
    for name in leftovers:               # a write that died before it was renamed into place
        drop(name + ".tmp")
    return removed


def build(corpus: "str | Path", labels: Iterable["str | Path"], out: "str | Path", *, combiner_path: "str | Path | None" = None,
          missing: str = "drop", traces_path: "str | Path | None" = None, thinker: Optional[str] = None,
          allow_undecontaminated: bool = False) -> Dict[str, Any]:
    if missing not in MISSING_POLICIES:
        raise TargetError(f"--missing must be one of {MISSING_POLICIES}")
    corpus_root, out_root = Path(corpus), Path(out)
    labels = list(labels)
    provenance = corpus_provenance(corpus_root, allow_undecontaminated=allow_undecontaminated)
    try:
        corpus_parts = S.corpus_files(corpus_root, require_manifest=True)
        label_files = sorted({f for path in labels for f in S.corpus_files(path, require_manifest=True)})
    except S.ManifestError as exc:
        raise TargetError(str(exc)) from exc
    if not corpus_parts or not all(p.is_file() for p in corpus_parts):
        raise TargetError(f"no corpus parquet parts found in {corpus_root}")
    if not label_files:
        raise TargetError("no teacher label parquet parts found")
    available = teacher_names(label_files)
    combiner = resolve_combiner(combiner_path, available)
    traces: Dict[str, Tuple[str, int, str, str]] = {}
    traces_digest = None
    if traces_path is not None:
        try:
            trace_files = S.corpus_files(traces_path, require_manifest=True)
            traces, thinker = load_traces(trace_files, thinker)
            traces_digest = _digest([(str(f.name), S.sha256_file(f)) for f in trace_files] + [("thinker", thinker or "")])
        except S.ManifestError as exc:
            raise TargetError(str(exc)) from exc
    labels_digest = _digest([(f.name, S.sha256_file(f)) for f in label_files])

    names = {part: _out_name(part, corpus_root) for part in corpus_parts}
    if len(set(names.values())) != len(names):
        raise TargetError("two corpus parts would produce the same output name")
    _check_out(out_root, [corpus_root, *(Path(path) for path in labels), *([Path(traces_path)] if traces_path is not None else [])])
    try:
        earlier = S.assert_manifest_owner(out_root, TOOL)         # refuse a manifest / report that another tool wrote, before anything is deleted
    except S.ManifestError as exc:
        raise TargetError(str(exc)) from exc
    _assert_report_is_ours(out_root)
    owned = _owned(out_root, earlier)              # also refuses a manifest that names a path: nothing has been touched yet
    alien = sorted(name for name in names.values() if (out_root / name).exists() and name not in owned)
    if alien:
        raise TargetError(f"{out_root} already holds {', '.join(alien)}, which build_targets did not write (no marker and not in its last manifest): it will "
                          "not overwrite it; move it away or use another --out")
    out_root.mkdir(parents=True, exist_ok=True)
    # a folder without a manifest is an unfinished build and the trainer refuses it: drop the old manifest and report before
    # touching anything, and write the new ones only at the end
    previous = S.claim_manifest(out_root, TOOL)
    (out_root / "report.json").unlink(missing_ok=True)
    removed = _prune(out_root, set(names.values()), previous)
    entries: List[Dict[str, Any]] = []
    index = None
    resumed = 0
    for part in corpus_parts:
        name = names[part]
        fingerprint = _digest([("v", str(FORMAT_VERSION)), ("corpus", S.sha256_file(part)), ("provenance", S.canonical_json(provenance)), ("labels", labels_digest),
                               ("combiner", combiner.sha256), ("missing", missing), ("traces", traces_digest or "")])
        target, marker = out_root / name, _marker(out_root, name)
        old = _read_marker(marker)
        if old and old.get("fingerprint") == fingerprint and target.is_file() and S.sha256_file(target) == old.get("sha256"):
            entries.append(old)
            resumed += 1
            continue
        marker.unlink(missing_ok=True)
        if index is None:
            index = index_labels(label_files)
        rows, stats = _build_part(part, index, combiner, missing, traces)
        info = S._write_part(target, rows, target_arrow_schema(), TARGET_COLUMNS)
        entry = {"name": name, "rows": info["rows"], "sha256": info["sha256"], "fingerprint": fingerprint, "stats": stats}
        S.write_manifest(marker, entry)
        entries.append(entry)
    manifest: Dict[str, Any] = {"schema_version": S.SCHEMA_VERSION, "kind": "targets", "tool": TOOL, "format_version": FORMAT_VERSION, "columns": list(TARGET_COLUMNS),
                                "corpus": provenance, "rows": sum(entry["rows"] for entry in entries), "parts": entries}
    S.write_manifest(out_root / S.MANIFEST_NAME, manifest)

    totals: Counter = Counter()
    missing_total: Counter = Counter()
    for info in manifest["parts"]:
        for key, value in info["stats"].items():
            if key == "missing_teacher":
                missing_total.update(value)
            else:
                totals[key] += value
    report = {
        "format_version": FORMAT_VERSION, "tool": TOOL, "corpus": provenance,
        "rows_in": totals["rows_in"], "rows_out": totals["rows_out"], "dropped_unlabeled": totals["dropped_unlabeled"],
        "questions": totals["questions"], "unlabeled_questions": totals["unlabeled_questions"],
        "teachers_seen": sorted(available), "missing_teacher": dict(sorted(missing_total.items())), "missing_policy": missing,
        "combiner_sha256": combiner.sha256, "parts": len(manifest["parts"]), "parts_resumed": resumed, "removed_stale": removed,
        "thinker": thinker if traces_path is not None else None,
        "reasoning_rows": totals["reasoning_rows"], "reasoning_truncated": totals["reasoning_truncated"], "reasoning_state_has_block": totals["reasoning_state_has_block"],
        "traces_unmatched": (len(traces) - totals["traces_matched"]) if traces_path is not None else 0,
    }
    S.write_manifest(out_root / "report.json", report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m decide.train.build_targets", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="the folder decontaminate left (SCHEMA.md, sections 3 and 11)")
    parser.add_argument("--labels", required=True, nargs="+", help="teacher-label folders with a manifest (seal_directory makes one) or parquet files (section 4)")
    parser.add_argument("--out", required=True, help="output folder (empty or the result of an earlier run of this command)")
    parser.add_argument("--combiner", help="combiner JSON (section 5); without it the labels must hold exactly one teacher")
    parser.add_argument("--missing", choices=MISSING_POLICIES, default="drop", help="a needed teacher has no label for a question (default: drop)")
    parser.add_argument("--traces", help="thinker traces (section 10): a folder with a manifest or a parquet file")
    parser.add_argument("--thinker", help="which thinker's traces to use when --traces holds several")
    parser.add_argument("--allow-undecontaminated", action="store_true",
                        help="accept a corpus that did not come out of decontaminate (a raw build, a hand-made folder, a parquet file); recorded in the targets")
    args = parser.parse_args(argv)
    report = build(args.corpus, args.labels, args.out, combiner_path=args.combiner, missing=args.missing, traces_path=args.traces, thinker=args.thinker,
                   allow_undecontaminated=args.allow_undecontaminated)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TargetError as error:
        sys.exit(f"error: {error}")
