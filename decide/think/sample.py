"""Deterministic stratified sample of the public knowledge rows, and the private pack that carries them.

Two outputs, both reproducible from the same inputs:

* `manifest.json` (safe to commit): seed, counts, and for each benchmark the run ids and payload
  hashes of the sampled rows, plus the sha256 of the pack content.  No question text.
* `pack.jsonl.gz` (PRIVATE: keep it out of git and out of public datasets, it holds GPQA, BBH and
  other benchmark text under their own terms): one line per sampled request with the state, the
  questions, the gold and the per-option probabilities of every decision model.

The sample is a function of (seed, benchmark, case id) only.  Each case gets a rank by
sha256("<seed>|case|<benchmark>|<case id>"); the sample of size n is the n lowest ranks, so a sample
of 40 is a prefix of a sample of 100, and changing the order of the input, or the models, never
changes it.  One request per case: GSM8K v0.3 has a 4-choice and a 10-choice request for the same
problem, and the bootstrap needs independent units.  The pack lists rows round-robin across
benchmarks, so any prefix of a run is still a stratified sample.

CLI:  python -m decide.think.sample build|verify ...  (see --help)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import common as C

ROWS_FILE = "selected-rows.jsonl.gz"
ADDED_FILE = "added-rows.jsonl.gz"
GSM8K_FILE = "gsm8k-rows.jsonl.gz"
EXCLUSIONS_FILE = "excluded-questions.json"
GSM8K_V3_PREFIX = "v0.3-gsm8k:"          # edition 0.3 replaces the old GSM8K rows by rows with this run id prefix
GSM8K_ID = 30


# ------------------------------------------------------------------------------------- the draw

@dataclass(frozen=True)
class Candidate:
    run_id: str
    catalog_id: int
    group_id: str
    track: str = ""
    payload_sha256: str = ""


@dataclass
class Draw:
    chosen: Dict[int, List[Candidate]] = field(default_factory=dict)
    population: Dict[int, int] = field(default_factory=dict)       # cases per benchmark (before eligibility)
    ineligible: Dict[int, int] = field(default_factory=dict)       # cases skipped (no eligible row) before the last sampled one
    missing: List[int] = field(default_factory=list)               # requested benchmarks with no candidate at all

    def total(self) -> int:
        return sum(len(v) for v in self.chosen.values())


def rank_hash(seed: int, kind: str, *parts: Any) -> str:
    return C.sha256_text("|".join([str(seed), kind, *map(str, parts)]))


def draw(candidates: Iterable[Candidate], *, per_benchmark: Optional[int] = C.PER_BENCHMARK, seed: int = C.SEED,
         benchmarks: Sequence[int] = C.DEFAULT_BENCHMARKS, one_per_case: bool = True,
         eligible: Optional[Callable[[Candidate], bool]] = None) -> Draw:
    """The sample: for each benchmark the `per_benchmark` lowest-ranked eligible cases (all of them when 0/None)."""
    wanted = set(int(b) for b in benchmarks)
    cases: Dict[int, Dict[str, List[Candidate]]] = {b: defaultdict(list) for b in wanted}
    for c in candidates:
        if c.catalog_id in wanted:
            cases[c.catalog_id][c.group_id if one_per_case else c.run_id].append(c)
    out = Draw()
    for b in sorted(wanted):
        groups = cases[b]
        if not groups:
            out.missing.append(b)
            continue
        out.population[b] = len(groups)
        ranked = sorted(groups, key=lambda g: (rank_hash(seed, "case", b, g), g))
        picked: List[Candidate] = []
        skipped = 0
        for g in ranked:
            rows = [r for r in groups[g] if eligible is None or eligible(r)]
            if not rows:
                skipped += 1
                continue
            picked.append(min(rows, key=lambda r: (rank_hash(seed, "row", b, r.run_id), r.run_id)))
            if per_benchmark and len(picked) >= per_benchmark:
                break
        out.chosen[b] = picked
        out.ineligible[b] = skipped
    return out


def interleave(d: Draw) -> List[Candidate]:
    """Round-robin across benchmarks (ascending id) by rank, so every prefix is stratified."""
    lists = [d.chosen[b] for b in sorted(d.chosen)]
    longest = max((len(x) for x in lists), default=0)
    return [lst[i] for i in range(longest) for lst in lists if i < len(lst)]


# ---------------------------------------------------------------------------------- suite rows

def candidate_of(row: dict) -> Optional[Candidate]:
    """None when the row is not a request the thinker can answer (only `choice` questions with a gold)."""
    e = row.get("_evaluation") or {}
    questions = row.get("questions") or {}
    expected = row.get("expected") or {}
    if not questions or any((q.get("type") or "choice") != "choice" for q in questions.values()):
        return None
    if any(expected.get(qid) is None for qid in questions):
        return None
    return Candidate(run_id=str(e["run_id"]), catalog_id=int(e["catalog_id"]), group_id=str(e.get("group_id", e["run_id"])),
                     track=str(e.get("track", "")), payload_sha256=str(e.get("payload_sha256", "")))


def gold_sets(row: dict) -> Dict[str, List[str]]:
    """qid -> accepted keys.  A one-question row with `scoring.accepted` (ChessBench ties) accepts all of them, as the kit does."""
    accepted = (row.get("scoring") or {}).get("accepted")
    questions = row["questions"]
    out: Dict[str, List[str]] = {}
    for qid in questions:
        if len(questions) == 1 and isinstance(accepted, list) and accepted:
            out[qid] = [str(a) for a in accepted]
        else:
            out[qid] = [str(row["expected"][qid])]
    return out


def suite_files(suite_dir: os.PathLike) -> List[Path]:
    d = Path(suite_dir)
    names = [ROWS_FILE, ADDED_FILE, GSM8K_FILE]
    files = [d / n for n in names if (d / n).exists()]
    if not (d / ROWS_FILE).exists():
        raise FileNotFoundError(f"{d / ROWS_FILE} not found: build the suite with the kit (`suite rebuild` + `suite import`) first")
    return files


def iter_suite_rows(suite_dir: os.PathLike, benchmarks: Optional[Iterable[int]] = None) -> Iterator[Tuple[str, dict]]:
    """(raw line, row) of edition 0.3 after the kit's exclusions and the GSM8K v0.3 replacement.

    Same rows the kit's `Suite.rows(apply_exclusions=True)` yields for these benchmarks.
    """
    d = Path(suite_dir)
    excluded = set()
    ex = d / EXCLUSIONS_FILE
    if ex.exists():
        excluded = set(json.loads(ex.read_text(encoding="utf-8")).get("rows", []))
    gsm_replaced = (d / GSM8K_FILE).exists()
    keep = None if benchmarks is None else {int(b) for b in benchmarks}
    for path in suite_files(d):
        with C.open_text(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                e = row["_evaluation"]
                n = int(e["catalog_id"])
                if keep is not None and n not in keep:
                    continue
                if e["run_id"] in excluded:
                    continue
                if gsm_replaced and n == GSM8K_ID and not str(e["run_id"]).startswith(GSM8K_V3_PREFIX):
                    continue
                yield line, row


# ------------------------------------------------------------------ the decision models' answers

_RUN_ID = re.compile(r'"run_id"\s*:\s*"([^"\\]*)"')


def decision_from_answers(answers: Any, questions: Dict[str, dict]) -> Optional[Dict[str, dict]]:
    """Validate one model's answers for a row and keep what the comparison needs, else None."""
    if not isinstance(answers, dict):
        return None
    out: Dict[str, dict] = {}
    for qid, q in questions.items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            return None
        probs = a.get("probabilities")
        keys = [str(k) for k in q["criteria"]]
        if not isinstance(probs, dict) or set(map(str, probs)) != set(keys):
            return None
        vals = {}
        for k in keys:
            v = probs.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 or v > 1:
                return None
            vals[k] = v
        choice = a.get("choice")
        out[qid] = {"choice": str(choice) if choice is not None else None, "probs": vals}
    return out


def scan_results(path: os.PathLike, wanted: Dict[str, dict], *, keep: Optional[Iterable[str]] = None) -> Dict[str, Optional[dict]]:
    """For each wanted run id found in a decision model's results file: its validated decision, or None if not usable.

    `wanted` maps run_id -> row (the row gives the questions and the payload hash to check against).  When `keep`
    is given only those ids hold a decision (the others hold True), which keeps memory flat in the first pass.
    """
    keep_set = None if keep is None else set(keep)
    found: Dict[str, Optional[dict]] = {}
    with C.open_text(path) as f:
        for line in f:
            m = _RUN_ID.search(line)
            if m is None or m.group(1) not in wanted:
                continue
            rid = m.group(1)
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            row = wanted[rid]
            ok = rec.get("status") == "ok"
            sha = rec.get("payload_sha256")
            if sha and row["_evaluation"].get("payload_sha256") and sha != row["_evaluation"]["payload_sha256"]:
                ok = False
            dec = decision_from_answers((rec.get("response") or {}).get("answers"), row["questions"]) if ok else None
            if dec is None:
                found[rid] = None
            elif keep_set is None or rid in keep_set:
                found[rid] = dec
            else:
                found[rid] = {}
    return found


# ----------------------------------------------------------------------------------- the pack

def pack_row(row: dict, decisions: Dict[str, Dict[str, dict]]) -> dict:
    e = row["_evaluation"]
    return {
        "id": e["run_id"],
        "benchmark": int(e["catalog_id"]),
        "track": e.get("track", ""),
        "group": e.get("group_id", ""),
        "payload_sha256": e.get("payload_sha256", ""),
        "state": row.get("state"),
        "questions": {qid: {"type": q.get("type", "choice"), "instructions": q.get("instructions", ""), "criteria": q["criteria"]}
                      for qid, q in row["questions"].items()},
        "gold": gold_sets(row),
        "decisions": {m: decisions[m] for m in sorted(decisions)},
    }


def pack_lines(rows: Iterable[dict]) -> Iterator[str]:
    for r in rows:
        yield json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_pack(path: os.PathLike, rows: Iterable[dict]) -> Tuple[int, str]:
    """Write the pack as gzip JSONL with a fixed gzip header; return (rows, sha256 of the uncompressed content)."""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    h = hashlib.sha256()
    n = 0
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz:
            for line in pack_lines(rows):
                b = line.encode("utf-8")
                gz.write(b)
                h.update(b)
                n += 1
    os.replace(tmp, p)
    return n, h.hexdigest()


def read_pack(path: os.PathLike) -> List[dict]:
    return list(C.iter_jsonl(path, complete_lines_only=False))


def pack_content_sha256(path: os.PathLike) -> str:
    h = hashlib.sha256()
    with (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")) as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(d: Draw, order: List[Candidate], *, seed: int, per_benchmark: Optional[int], one_per_case: bool,
                   models: Sequence[str], pack_rows: int, pack_sha256: str, suite_hashes: Dict[str, Dict[str, str]],
                   results_hashes: Dict[str, Dict[str, str]], skipped: Dict[str, int]) -> dict:
    benchmarks = {}
    for b in sorted(d.chosen):
        lst = d.chosen[b]
        benchmarks[str(b)] = {
            "name": C.benchmark_name(b),
            "population_cases": d.population.get(b, 0),
            "skipped_ineligible_cases": d.ineligible.get(b, 0),
            "n": len(lst),
            "run_ids": [c.run_id for c in lst],
            "payload_sha256": [c.payload_sha256 for c in lst],
        }
    return {
        "format": C.MANIFEST_FORMAT,
        "seed": seed,
        "per_benchmark": per_benchmark or 0,
        "one_per_case": one_per_case,
        "benchmarks": benchmarks,
        "missing_benchmarks": sorted(d.missing),
        "decision_models": sorted(models),
        "order": "round-robin by rank across benchmarks in ascending catalog id",
        "pack": {"format": C.PACK_FORMAT, "rows": pack_rows, "content_sha256": pack_sha256},
        "inputs": {"suite_files": dict(sorted(suite_hashes.items())), "results": dict(sorted(results_hashes.items()))},
        "skipped": dict(sorted(skipped.items())),
        "note": "ids and hashes only; the pack with the question text is private (GPQA, BBH canary, HLE terms)",
        "use": C.EVAL_ONLY_NOTICE,
    }


def build(suite_dir: os.PathLike, results: Dict[str, os.PathLike], out_dir: os.PathLike, *, per_benchmark: Optional[int] = C.PER_BENCHMARK,
          seed: int = C.SEED, benchmarks: Sequence[int] = C.DEFAULT_BENCHMARKS, one_per_case: bool = True,
          log: Callable[[str], None] = lambda s: None) -> dict:
    """Draw the sample from the suite, attach the decision models' answers, write pack.jsonl.gz and manifest.json."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not results:
        raise ValueError("at least one decision model's results file is needed")
    rows: Dict[str, dict] = {}
    cands: List[Candidate] = []
    skipped = defaultdict(int)
    for _line, row in iter_suite_rows(suite_dir, benchmarks):
        c = candidate_of(row)
        if c is None:
            skipped["unsupported_or_ungolded_rows"] += 1
            continue
        rows[c.run_id] = row
        cands.append(c)
    log(f"suite: {len(cands)} candidate requests in {time.time() - t0:.0f}s")
    usable: Dict[str, set] = {}
    for model, path in sorted(results.items()):
        found = scan_results(path, rows, keep=())
        usable[model] = {rid for rid, v in found.items() if v is not None}
        log(f"results {model}: {len(usable[model])} usable of {len(found)} matched rows")
    ok_everywhere = set.intersection(*usable.values()) if usable else set()
    d = draw(cands, per_benchmark=per_benchmark, seed=seed, benchmarks=benchmarks, one_per_case=one_per_case,
             eligible=lambda c: c.run_id in ok_everywhere)
    order = interleave(d)
    chosen_ids = {c.run_id for c in order}
    decisions: Dict[str, Dict[str, Dict[str, dict]]] = {rid: {} for rid in chosen_ids}
    for model, path in sorted(results.items()):
        got = scan_results(path, {rid: rows[rid] for rid in chosen_ids})
        for rid in chosen_ids:
            dec = got.get(rid)
            if not dec:
                raise RuntimeError(f"{model} has no usable answer for {rid} (should have been filtered)")
            decisions[rid][model] = dec
    pack = (pack_row(rows[c.run_id], decisions[c.run_id]) for c in order)
    n, content_sha = write_pack(out / "pack.jsonl.gz", pack)
    suite_hashes = {p.name: C.file_digests(p) for p in suite_files(suite_dir)}
    ex = Path(suite_dir) / EXCLUSIONS_FILE
    if ex.exists():
        suite_hashes[ex.name] = C.file_digests(ex)
    results_hashes = {m: C.file_digests(p) for m, p in results.items()}
    manifest = build_manifest(d, order, seed=seed, per_benchmark=per_benchmark, one_per_case=one_per_case, models=list(results),
                              pack_rows=n, pack_sha256=content_sha, suite_hashes=suite_hashes, results_hashes=results_hashes,
                              skipped=dict(skipped))
    manifest_sha = C.write_json(out / "manifest.json", manifest)
    log(f"pack: {n} rows, content sha256 {content_sha[:12]}..., manifest sha256 {manifest_sha[:12]}... in {time.time() - t0:.0f}s")
    return {"rows": n, "pack_content_sha256": content_sha, "manifest_sha256": manifest_sha, "per_benchmark": {b: len(v) for b, v in d.chosen.items()},
            "missing_benchmarks": d.missing, "out": str(out)}


def verify(manifest_path: os.PathLike, pack_path: os.PathLike) -> List[str]:
    """Problems found when checking a pack against a manifest (empty list = the pack is the manifest's sample)."""
    problems: List[str] = []
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if m.get("format") != C.MANIFEST_FORMAT:
        problems.append(f"manifest format is {m.get('format')!r}, expected {C.MANIFEST_FORMAT!r}")
    rows = read_pack(pack_path)
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        problems.append("pack has duplicate ids")
    expected_order: List[str] = []
    per = {int(b): v for b, v in m.get("benchmarks", {}).items()}
    longest = max((len(v["run_ids"]) for v in per.values()), default=0)
    for i in range(longest):
        for b in sorted(per):
            if i < len(per[b]["run_ids"]):
                expected_order.append(per[b]["run_ids"][i])
    if [r["id"] for r in rows] != expected_order:
        problems.append(f"pack ids/order differ from the manifest ({len(rows)} rows in the pack, {len(expected_order)} in the manifest)")
    for b, v in per.items():
        for rid, sha in zip(v["run_ids"], v["payload_sha256"]):
            r = by_id.get(rid)
            if r is not None and sha and r.get("payload_sha256") != sha:
                problems.append(f"{rid}: payload sha256 differs from the manifest")
    if pack_content_sha256(pack_path) != m.get("pack", {}).get("content_sha256"):
        problems.append("pack content sha256 differs from the manifest")
    models = set(m.get("decision_models", []))
    for r in rows:
        if set(r.get("decisions", {})) != models:
            problems.append(f"{r['id']}: decision models differ from the manifest")
            break
    return problems


# ------------------------------------------------------------------------------------------ CLI

def _parse_results(items: Sequence[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        name, sep, path = it.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--results expects NAME=PATH, got {it!r}")
        out[name] = path
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m decide.think.sample", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="draw the sample and write pack.jsonl.gz + manifest.json")
    b.add_argument("--suite", required=True, help="directory with selected-rows.jsonl.gz, added-rows.jsonl.gz, gsm8k-rows.jsonl.gz, excluded-questions.json")
    b.add_argument("--results", action="append", default=[], metavar="NAME=PATH", help="a decision model's per-row results (repeat); .jsonl or .jsonl.gz")
    b.add_argument("--out", required=True)
    b.add_argument("--per-benchmark", type=int, default=C.PER_BENCHMARK, help="rows per benchmark; 0 = every row")
    b.add_argument("--seed", type=int, default=C.SEED)
    b.add_argument("--benchmarks", default=",".join(map(str, C.DEFAULT_BENCHMARKS)), help="comma-separated catalog ids")
    b.add_argument("--all-requests", action="store_true", help="do not reduce to one request per case (validation against the kit only)")
    v = sub.add_parser("verify", help="check a pack against a manifest")
    v.add_argument("--manifest", required=True)
    v.add_argument("--pack", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        info = build(args.suite, _parse_results(args.results), args.out, per_benchmark=args.per_benchmark or None, seed=args.seed,
                     benchmarks=[int(x) for x in args.benchmarks.split(",") if x], one_per_case=not args.all_requests,
                     log=lambda s: print(f"[sample] {s}", flush=True))
        print(json.dumps(info, indent=2, sort_keys=True))
        if info["missing_benchmarks"]:
            print(f"[sample] benchmarks without rows (HLE is gated: it needs a suite built with an HF token): {info['missing_benchmarks']}")
        return 0
    problems = verify(args.manifest, args.pack)
    for p in problems:
        print(f"[verify] {p}")
    print("[verify] OK: the pack is the manifest's sample" if not problems else f"[verify] {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
