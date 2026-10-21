"""F1 fidelity: do our answers match the per-row results published for d3 and d3-mini?

The question behind gate G0: when the same request goes through our loader (4-bit on a T4, or fp16 for the 4B model)
instead of the authors' bf16 runtime, does the model still decide the same way?  Two public datasets hold the authors'
per-row results of the complete public suite: `vllm-sr/d3-mini-decision-index` (4B) and `vllm-sr/d3-decision-index` (27B).
Their rows carry `payload_sha256`, the hash of the request they answered, so a comparison can prove it is looking at the
same request.  The requests themselves (the suite rows) are not in those datasets; they come from the user's own copy of
the suite or from a rebuild, and the sample must hold at least `min_rows` of them.

What is compared, per question: the argmax option (`noul`: which side of one half), over every row that is `ok` on both
sides with the same payload hash.  The headline is the field (question) agreement; the row agreement (every question of a
request equal) is reported next to it, with a cluster bootstrap interval over groups.  Disagreements where the published
top two options were within `near_tie` of each other are counted apart: a coin-flip is not a distortion.

Nothing here imports torch.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from decide.sd import evaluate
from decide.sd import sample as sample_mod
from decide.sd.remote import HttpTransport, RemoteError

ENDPOINT = "https://huggingface.co"
RESULTS_FILE = "results.jsonl.gz"
SUMS_FILE = "SHA256SUMS"
REPORT_NAME = "fidelity.json"
KEEP = ("run_id", "catalog_id", "group_id", "status", "payload_sha256")


class FidelityError(RuntimeError):
    pass


# --------------------------------------------------------------------------- the published files

def published_url(dataset: str, run: str, filename: str = RESULTS_FILE, *, endpoint: str = ENDPOINT, revision: str = "main") -> str:
    return f"{endpoint.rstrip('/')}/datasets/{dataset}/resolve/{revision}/runs/{run}/{filename}"


def parse_sha256sums(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(\S.*)$", line.strip())
        if m:
            out[m.group(2).strip()] = m.group(1).lower()
    return out


def sha256_file(path: Union[str, os.PathLike]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Union[str, os.PathLike], sums_text: str, name: Optional[str] = None) -> None:
    sums = parse_sha256sums(sums_text)
    key = name or Path(path).name
    if key not in sums:
        raise FidelityError(f"no checksum for {key} in the SHA256SUMS file")
    got = sha256_file(path)
    if got != sums[key]:
        raise FidelityError(f"{key}: sha256 {got} does not match the published {sums[key]}; the download is damaged or the dataset changed")


def fetch_published(dataset: str, run: str, dest_dir: Union[str, os.PathLike], *, token: Optional[str] = None, endpoint: str = ENDPOINT,
                    revision: str = "main", sleep: Callable[[float], None] = time.sleep, verify: bool = True,
                    transport: Any = None) -> Path:
    """Download `runs/<run>/results.jsonl.gz` of a public results dataset (about 47 MB) and check it against its SHA256SUMS."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / RESULTS_FILE
    http = transport or HttpTransport(token=token, sleep=sleep)
    sums_text: Optional[str] = None
    if verify:
        try:
            sums_text = http.get(published_url(dataset, run, SUMS_FILE, endpoint=endpoint, revision=revision)).decode("utf-8")
        except RemoteError as exc:
            raise FidelityError(f"cannot read SHA256SUMS of {dataset} run {run}: {exc}") from exc
        if target.exists():
            try:
                verify_sha256(target, sums_text, RESULTS_FILE)
                return target
            except FidelityError:
                pass                                              # a stale or damaged copy: download again
    elif target.exists():
        return target
    try:
        blob = http.get(published_url(dataset, run, RESULTS_FILE, endpoint=endpoint, revision=revision))
    except RemoteError as exc:
        raise FidelityError(f"cannot download {dataset} run {run}: {exc}") from exc
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(blob)
    try:
        if sums_text is not None:
            verify_sha256(tmp, sums_text, RESULTS_FILE)
    except FidelityError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, target)
    return target


def load_published(path: Union[str, os.PathLike], wanted: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """run_id -> the few fields the comparison reads.  With `wanted`, other rows are dropped while streaming."""
    keep = None if wanted is None else set(wanted)
    p = Path(path)
    out: Dict[str, Dict[str, Any]] = {}
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as handle:           # type: ignore[operator]
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise FidelityError(f"{p.name} line {number}: not valid JSON ({exc})") from exc
            run_id = row.get("run_id")
            if keep is not None and run_id not in keep:
                continue
            slim = {k: row.get(k) for k in KEEP}
            response = row.get("response")
            slim["response"] = {"answers": response.get("answers", {})} if isinstance(response, Mapping) else None
            out[run_id] = slim
    return out


# --------------------------------------------------------------------------- the comparison

def _view(answer: Any) -> Optional[Tuple[str, List[str], List[float], str]]:
    """(type, option keys, probabilities, argmax key) of one answer, or None when it is not a well-formed answer."""
    if not isinstance(answer, Mapping):
        return None
    try:
        if answer.get("type") == "noul":
            p = float(answer["noul"])
            return "noul", ["false", "true"], [1.0 - p, p], "true" if p >= 0.5 else "false"
        if answer.get("type") == "choice":
            keys = list(answer["probabilities"])
            probs = [float(answer["probabilities"][k]) for k in keys]
            if not keys:
                return None
            best = max(range(len(keys)), key=lambda i: probs[i])
            picked = answer.get("choice")
            return "choice", keys, probs, picked if picked in keys else keys[best]
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _margin(kind: str, probs: Sequence[float]) -> float:
    if kind == "noul":
        return abs(probs[1] - 0.5) * 2.0
    top = sorted(probs, reverse=True)
    return top[0] - top[1] if len(top) > 1 else 1.0


def compare(ours: Mapping[str, Mapping[str, Any]], published: Mapping[str, Mapping[str, Any]], *, near_tie: float = 0.02,
            replicates: int = 500, seed: int = 0, level: float = 0.95, examples: int = 20) -> Dict[str, Any]:
    """Agreement of the argmax of every question between two sets of result rows (see the module docstring)."""
    counters = {"rows": 0, "agree_rows": 0, "fields": 0, "agree_fields": 0, "not_ok": 0, "missing_in_published": 0,
                "payload_mismatch": 0, "structure_mismatch": 0, "disagree_near_tie": 0, "disagree_clear": 0}
    per_benchmark: Dict[str, Dict[str, int]] = {}
    per_group: Dict[Any, List[int]] = {}
    sum_abs = sum_tv = 0.0
    max_abs = 0.0
    compared_options = 0
    listed: List[Dict[str, Any]] = []
    for run_id, mine in ours.items():
        theirs = published.get(run_id)
        if theirs is None:
            counters["missing_in_published"] += 1
            continue
        if mine.get("status") != "ok" or theirs.get("status") != "ok":
            counters["not_ok"] += 1
            continue
        if mine.get("payload_sha256") and theirs.get("payload_sha256") and mine["payload_sha256"] != theirs["payload_sha256"]:
            counters["payload_mismatch"] += 1
            continue
        mine_answers = (mine.get("response") or {}).get("answers") or {}
        their_answers = (theirs.get("response") or {}).get("answers") or {}
        catalog = str(mine.get("catalog_id", theirs.get("catalog_id")))
        group = mine.get("group_id", theirs.get("group_id", run_id))
        bench = per_benchmark.setdefault(catalog, {"rows": 0, "fields": 0, "agree_fields": 0})
        bench["rows"] += 1
        counters["rows"] += 1
        row_ok = True
        for qid in sorted(set(mine_answers) | set(their_answers)):
            a, b = _view(mine_answers.get(qid)), _view(their_answers.get(qid))
            counters["fields"] += 1
            bench["fields"] += 1
            tally = per_group.setdefault(group, [0, 0])
            tally[1] += 1
            if a is None or b is None or a[0] != b[0] or a[1] != b[1]:
                counters["structure_mismatch"] += 1
                counters["disagree_clear"] += 1
                row_ok = False
                continue
            diffs = [abs(x - y) for x, y in zip(a[2], b[2])]
            sum_abs += sum(diffs) / len(diffs)
            sum_tv += 0.5 * sum(diffs)
            max_abs = max(max_abs, max(diffs))
            compared_options += 1
            if a[3] == b[3]:
                counters["agree_fields"] += 1
                bench["agree_fields"] += 1
                tally[0] += 1
                continue
            row_ok = False
            margin = _margin(b[0], b[2])
            counters["disagree_near_tie" if margin < near_tie else "disagree_clear"] += 1
            if len(listed) < examples:
                listed.append({"run_id": run_id, "question_id": qid, "catalog_id": catalog, "ours": a[3], "published": b[3],
                               "p_ours": max(a[2]) if a[0] == "choice" else a[2][1], "p_published": max(b[2]) if b[0] == "choice" else b[2][1],
                               "published_margin": margin})
        counters["agree_rows"] += int(row_ok)
    fields = counters["fields"]
    rows = counters["rows"]
    ci = None
    if fields and replicates > 0:
        tallies = list(per_group.values())
        rng = random.Random(seed)
        values = []
        for _ in range(replicates):
            agree = total = 0
            for _g in tallies:
                pick = tallies[rng.randrange(len(tallies))]
                agree += pick[0]
                total += pick[1]
            values.append(agree / total if total else 0.0)
        tail = (1.0 - level) / 2.0
        ci = [evaluate.percentile(values, tail), evaluate.percentile(values, 1.0 - tail)]
    return {
        **counters,
        "missing_in_ours": len(set(published) - set(ours)),
        "field_agreement": counters["agree_fields"] / fields if fields else None,
        "row_agreement": counters["agree_rows"] / rows if rows else None,
        "ci": ci, "ci_level": level, "replicates": replicates, "seed": seed, "near_tie": near_tie,
        "prob": {"mean_abs": sum_abs / compared_options if compared_options else None, "max_abs": max_abs if compared_options else None,
                 "mean_tv": sum_tv / compared_options if compared_options else None},
        "per_benchmark": {k: {**v, "agreement": v["agree_fields"] / v["fields"] if v["fields"] else None} for k, v in sorted(per_benchmark.items())},
        "disagreements": listed,
    }


def gate(result: Mapping[str, Any], *, threshold: float, min_rows: int = 500) -> Dict[str, Any]:
    """G0: field agreement at or above `threshold` over at least `min_rows` rows.  `firm` = the interval's lower end clears it too."""
    rows, value, ci = int(result.get("rows") or 0), result.get("field_agreement"), result.get("ci")
    out: Dict[str, Any] = {"value": value, "threshold": threshold, "rows": rows, "min_rows": min_rows, "ci": ci, "firm": False}
    if rows < min_rows or value is None:
        out.update(status="insufficient", reason=f"{rows} comparable rows; the gate needs at least {min_rows}")
        return out
    out["status"] = "pass" if value >= threshold else "fail"
    out["firm"] = bool(ci and ci[0] >= threshold)
    out["reason"] = (f"field agreement {value:.4f} {'>=' if value >= threshold else '<'} {threshold} over {rows} rows"
                     + (f" (95% interval {ci[0]:.4f}..{ci[1]:.4f})" if ci else ""))
    return out


# --------------------------------------------------------------------------- one check, end to end

def _row_of(item: evaluate.Item) -> Dict[str, Any]:
    row: Dict[str, Any] = {"_evaluation": item.meta, "state": item.inputs["state"], "questions": item.inputs["questions"]}
    if item.gold is not None:
        row["expected"] = item.gold
    return row


def check_fidelity(engine: Any, requests_path: Union[str, os.PathLike], published_path: Union[str, os.PathLike],
                   out_dir: Union[str, os.PathLike], *, threshold: float, min_rows: int = 500, limit: Optional[int] = None,
                   budget: Any = None, state: Any = None, chunk: int = 8, log: Optional[Callable[[str], None]] = print,
                   name: Optional[str] = None, near_tie: float = 0.02, replicates: int = 500, seed: int = 0) -> Dict[str, Any]:
    """Run the requests that the published file also answered, compare, apply the gate, write `fidelity.json`."""
    say = log or (lambda text: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = evaluate.read_items(requests_path)
    published = load_published(published_path, {i.run_id for i in items})
    usable = []
    skipped = {"not_published": 0, "published_not_ok": 0, "payload_mismatch": 0}
    for item in items:
        theirs = published.get(item.run_id)
        if theirs is None:
            skipped["not_published"] += 1
        elif theirs.get("status") != "ok":
            skipped["published_not_ok"] += 1
        elif item.meta.get("payload_sha256") and theirs.get("payload_sha256") and item.meta["payload_sha256"] != theirs["payload_sha256"]:
            skipped["payload_mismatch"] += 1
        else:
            usable.append(item)
    if not usable:
        raise FidelityError(f"none of the {len(items)} requests has a published answer with the same payload: {skipped}")
    if limit:
        usable = usable[:limit]
    say(f"[fidelity] {len(usable)} of {len(items)} requests have a published answer ({skipped})")
    subset = out / "requests.jsonl.gz"
    with gzip.open(subset, "wt", encoding="utf-8") as handle:
        for item in usable:
            handle.write(json.dumps(_row_of(item), ensure_ascii=False) + "\n")
    run = evaluate.run_requests(engine, subset, out / "run", budget=budget, resume=True, chunk=chunk, state=state,
                                state_prefix=f"fidelity/{name or out.name}", log=log)
    result = compare(evaluate.read_results(out / "run"), {i.run_id: published[i.run_id] for i in usable}, near_tie=near_tie,
                     replicates=replicates, seed=seed)
    report = {"name": name or out.name, "complete": run["status"] == "complete", "run": {k: run[k] for k in ("rows", "total", "tokens", "seconds", "status", "counts")},
              "requests": {"in_file": len(items), "compared_candidates": len(usable), "skipped": skipped},
              "engine": dict(getattr(engine, "provenance", {}) or {}), "compare": result,
              "gate": gate(result, threshold=threshold, min_rows=min_rows), "published": Path(published_path).name,
              "note": ("Agreement is measured per question (argmax; noul: side of 0.5) over rows that are ok on both sides with the same payload hash. "
                       "The published results come from the authors' bf16 runtime on other hardware, so some disagreement is expected even from a perfect port.")}
    evaluate._atomic_json(out / REPORT_NAME, report)
    say(f"[fidelity] {report['gate']['status']}: {report['gate']['reason']}")
    return report


def common_sample(rows: Sequence[Mapping[str, Any]], published: Mapping[str, Mapping[str, Any]], *, per_benchmark: int, seed: int,
                  max_proxy_tokens: Optional[int] = None, edition: Optional[str] = None) -> sample_mod.Selection:
    """A stratified sample drawn only from the groups whose rows the published run answered with the same payload."""
    ok = []
    by_group: Dict[Tuple[Any, Any], List[Mapping[str, Any]]] = {}
    for row in rows:
        e = row["_evaluation"]
        by_group.setdefault((e["catalog_id"], e["group_id"]), []).append(row)
    for members in by_group.values():
        good = True
        for row in members:
            theirs = published.get(row["_evaluation"]["run_id"])
            mine = row["_evaluation"].get("payload_sha256")
            if theirs is None or theirs.get("status") != "ok" or (mine and theirs.get("payload_sha256") and mine != theirs["payload_sha256"]):
                good = False
                break
        if good:
            ok.extend(members)
    return sample_mod.select(ok, per_benchmark=per_benchmark, seed=seed, max_proxy_tokens=max_proxy_tokens, edition=edition)
