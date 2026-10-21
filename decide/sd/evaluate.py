"""Run requests through an engine with resume and a time budget, and score the result with the scoring kit itself.

`run_requests` is the loop every stage uses (smoke test, fidelity, merge sweep, calibration, teacher labels):

* requests come from a `.jsonl`, `.jsonl.gz` or `.parquet` file (kit rows, or plain `{run_id|example_id, request}` rows);
* the engine is asked for a chunk of requests at a time (`engine.batch`; `engine(**request)` for engines without it);
* every finished request is appended to `results.jsonl` in the kit's row format (so the kit's scorer and `upload_run`
  read it unchanged) and its raw per-view logits to `raw.jsonl`, then the chunk is flushed to disk and, when a `State`
  is given, mirrored to the Hub; a crash, a preemption or the end of the session loses at most one chunk;
* calling it again continues where the files stop (rows with status `error` are tried again, `unsupported` are final);
* a `Budget` is asked before each chunk whether the chunk still fits, so a session ends with a clean stop, not a kill.

The scoring helpers call the kit's own functions (`benchmark_summary`, `score_panel`, `index02`) on the sampled rows.
Nothing in here re-implements a metric.  Benchmarks that are not in the sample are dropped and the area weights are
renormalised over what is there, so the number is a sample index: good for comparing variants on the same sample,
not the board's public score and not the Full score.

Nothing here imports torch.
"""
from __future__ import annotations

import collections
import copy
import gzip
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from decide.sd import calibrate
from decide.sd import sample as sample_mod
from decide.sd.kitapi import NativeAbstention, Unsupported, validate

RESULTS_NAME = "results.jsonl"
RAW_NAME = "raw.jsonl"
TABLE_NAME = "results.parquet"
STATUS_NAME = "status.json"
ENVIRONMENT_NAME = "environment.json"
METRICS_NAME = "metrics.json"
PLAN_NAME = "plan.json"
SUBSET_DIR = "subset"
JSONL_SUFFIXES = (".jsonl", ".jsonl.gz", ".ndjson", ".ndjson.gz")
META_KEYS = ("example_id", "source", "license", "split", "lang", "catalog_id", "group_id", "track", "proxy_tokens", "dataset")
DEVICE_ERRORS = ("OutOfMemoryError", "AcceleratorError")
DEVICE_MARKERS = ("device-side assert", "CUDA error", "CUDA out of memory", "HIP error")
NOTE = ("Sample index: computed with the scoring kit's own scorer on the sampled rows only; benchmarks that are not in the "
        "sample are dropped and the area weights renormalised. Use it to compare variants on the same sample. It is not the "
        "board's public score and not the Full score.")


class EvaluateError(RuntimeError):
    pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# --------------------------------------------------------------------------- requests

@dataclass(frozen=True)
class Item:
    """One request: what the engine gets (`inputs`), what the result row carries over (`meta`), the gold labels if known."""

    run_id: str
    inputs: Dict[str, Any]
    meta: Dict[str, Any]
    gold: Optional[Dict[str, Any]] = None

    @property
    def proxy_tokens(self) -> int:
        return int(self.meta.get("proxy_tokens") or 0)

    def tokens(self) -> int:
        """Proxy tokens when the row has them, else a rough count (a third of the JSON characters)."""
        return self.proxy_tokens or max(1, len(json.dumps(self.inputs, ensure_ascii=False)) // 3)


def _json_field(value: Any, where: str, what: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as exc:
            raise EvaluateError(f"{where}: {what} is not valid JSON ({exc})") from exc
    return value


def item_from_row(row: Any, where: str = "row") -> Item:
    if not isinstance(row, Mapping):
        raise EvaluateError(f"{where}: a request row is a JSON object")
    marker = row.get("_evaluation")
    if isinstance(marker, Mapping):                              # a row of the Decision Index suite
        meta = dict(marker)
        run_id = meta.get("run_id")
        request = {k: row[k] for k in ("state", "questions", "images") if k in row}
        gold = row.get("expected")
    else:                                                        # {run_id | example_id, request | state + questions, gold}
        run_id = row.get("run_id", row.get("example_id"))
        request = row.get("request")
        if request is None:
            request = {k: row[k] for k in ("state", "questions", "images") if k in row}
            if "questions" in request:
                request["questions"] = _json_field(request["questions"], where, "questions")
        else:
            request = _json_field(request, where, "request")
        gold = _json_field(row.get("gold"), where, "gold")
        meta = {"run_id": run_id, **{k: row[k] for k in META_KEYS if k in row and row[k] is not None}}
    if not isinstance(run_id, str) or not run_id:
        raise EvaluateError(f"{where}: no run_id (or example_id)")
    questions = request.get("questions") if isinstance(request, Mapping) else None
    if not isinstance(questions, Mapping) or not questions or "state" not in request:
        raise EvaluateError(f"{where} ({run_id}): the request needs 'state' and a non-empty 'questions' object")
    if gold is not None and not isinstance(gold, Mapping):
        raise EvaluateError(f"{where} ({run_id}): gold must map question ids to labels")
    return Item(run_id, dict(request), meta, dict(gold) if gold is not None else None)


def items_from_rows(rows: Iterable[Mapping[str, Any]]) -> List[Item]:
    return [item_from_row(row, f"row {i + 1}") for i, row in enumerate(rows)]


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.name.lower().endswith(".gz") else open(path, "r", encoding="utf-8")


def iter_items(path: Union[str, os.PathLike]) -> Iterator[Item]:
    """The requests of a file, in file order, validated one by one."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(JSONL_SUFFIXES):
        with _open_text(p) as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise EvaluateError(f"{p.name} line {number}: not valid JSON ({exc})") from exc
                yield item_from_row(row, f"{p.name} line {number}")
    elif name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise EvaluateError("reading .parquet requests needs pyarrow (pip install pyarrow)") from exc
        number = 0
        for batch in pq.ParquetFile(str(p)).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                number += 1
                yield item_from_row(row, f"{p.name} row {number}")
    else:
        raise EvaluateError(f"{p.name}: unsupported suffix (use .jsonl, .jsonl.gz or .parquet)")


def read_items(path: Union[str, os.PathLike]) -> List[Item]:
    """All requests of a file; a run id that appears twice is refused (it would be counted twice)."""
    items, seen = [], set()
    for item in iter_items(path):
        if item.run_id in seen:
            raise EvaluateError(f"duplicate run_id {item.run_id!r} in {Path(path).name}")
        seen.add(item.run_id)
        items.append(item)
    return items


# --------------------------------------------------------------------------- files of a run

def _paths(out_path: Union[str, os.PathLike]) -> Tuple[Path, Path]:
    p = Path(out_path)
    if p.suffix.lower() == ".parquet":
        return p.parent, p
    return p, p / TABLE_NAME


def _drop_partial_tail(path: Path) -> None:
    """Cut the file after its last newline: a row half written when the process died is not a row."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
        end, block = size, 1 << 16
        while end > 0:
            start = max(0, end - block)
            handle.seek(start)
            chunk = handle.read(end - start)
            cut = chunk.rfind(b"\n")
            if cut >= 0:
                handle.truncate(start + cut + 1)
                return
            end = start
        handle.truncate(0)


def _read_rows(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise EvaluateError(f"{path.name} line {number}: not valid JSON ({exc}); the file is damaged") from exc


def read_results(run_dir: Union[str, os.PathLike]) -> Dict[str, Dict[str, Any]]:
    """run_id -> its last result row (a retried request has two rows; the later one counts)."""
    path = Path(run_dir)
    path = path / RESULTS_NAME if path.is_dir() else path
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_rows(path):
        out[row["run_id"]] = row
    return out


def read_raw(run_dir: Union[str, os.PathLike]) -> Dict[str, Dict[str, Any]]:
    path = Path(run_dir)
    path = path / RAW_NAME if path.is_dir() else path
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_rows(path):
        out[row["run_id"]] = row
    return out


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _append(path: Path, lines: Sequence[str]) -> None:
    if not lines:
        return
    with open(path, "ab") as handle:
        handle.write(("\n".join(lines) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _is_device_error(exc: BaseException) -> bool:
    return type(exc).__name__ in DEVICE_ERRORS or any(m in str(exc) for m in DEVICE_MARKERS)


# --------------------------------------------------------------------------- the loop

def _call(engine: Any, items: Sequence[Item]) -> List[Any]:
    """One (response, raw) or exception per item.  Only a device error escapes (as EvaluateError)."""
    batch = getattr(engine, "batch", None)

    def one(item: Item) -> Any:
        try:
            if callable(batch):
                return batch([item.inputs])[0]
            return engine(**item.inputs)
        except Exception as exc:                                 # noqa: BLE001 - one bad request must not stop the rest
            if _is_device_error(exc):
                raise EvaluateError(f"device error ({type(exc).__name__}: {exc}); stop before the accelerator context is reused") from exc
            return exc

    if not callable(batch):
        return [one(item) for item in items]
    try:
        got = list(batch([item.inputs for item in items]))
    except Exception as exc:                                     # noqa: BLE001
        if _is_device_error(exc):
            raise EvaluateError(f"device error ({type(exc).__name__}: {exc}); stop before the accelerator context is reused") from exc
        if len(items) == 1:
            return [exc]
        return [one(item) for item in items]                     # find the request that broke the call
    if len(got) != len(items):
        raise EvaluateError(f"engine.batch returned {len(got)} results for {len(items)} requests")
    return got


def _row_from(item: Item, got: Any, engine_name: str, started: str, finished: str, wall_ms: float, size: int) -> Tuple[Dict[str, Any], Optional[str]]:
    """(result row, raw line or None)."""
    row: Dict[str, Any] = {**item.meta, "started_utc": started, "engine": engine_name}
    raw_line: Optional[str] = None
    if isinstance(got, NativeAbstention):
        row.update(status="abstained", error=str(got))
    elif isinstance(got, Unsupported):
        row.update(status="unsupported", error=str(got))
    elif isinstance(got, BaseException):
        row.update(status="error", error=str(got), exception=type(got).__name__)
    else:
        try:
            response, raw = got
            validate(item.inputs["questions"], response)
            line = _dumps({**row, "status": "ok", "response": response})            # fails here on NaN, not later
            if isinstance(raw, Mapping):
                raw_line = _dumps({"run_id": item.run_id, **raw})
            row.update(status="ok", response=response)
            del line
        except Exception as exc:                                 # noqa: BLE001 - an invalid answer is a result, not a crash
            raw_line = None
            row.pop("response", None)
            row.update(status="error", error=f"{type(exc).__name__}: {exc}", exception=type(exc).__name__)
    row.update(completed_utc=finished, total_wall_ms=wall_ms, model_request_wall_ms=wall_ms, batch_size=size)
    return row, raw_line


def run_requests(engine: Any, requests_path: Union[str, os.PathLike], out_path: Union[str, os.PathLike], *,
                 budget: Any = None, resume: bool = True, chunk: int = 16, limit: Optional[int] = None,
                 log: Optional[Callable[[str], None]] = print, state: Any = None, state_prefix: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic, table: bool = True) -> Dict[str, Any]:
    """Answer every request of `requests_path` that has no final result yet.  See the module docstring.

    Returns `{"rows": new rows, "total": rows with a result, "tokens": input tokens of the new rows, "seconds", "status":
    "complete" | "budget" | "limit", "counts": statuses of the new rows, "skipped": rows kept from before, "files": {...}}`.
    """
    if chunk < 1:
        raise EvaluateError("chunk must be at least 1")
    say = log or (lambda text: None)
    out_dir, table_path = _paths(out_path)
    items = read_items(requests_path)                            # refuses duplicates before any work is done
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, raw_path = out_dir / RESULTS_NAME, out_dir / RAW_NAME
    prefix = (state_prefix or f"runs/{out_dir.name}").strip("/")
    if state is not None:
        state.track_file(results_path, f"{prefix}/{RESULTS_NAME}")
        state.track_file(raw_path, f"{prefix}/{RAW_NAME}")
        if resume:
            restored = state.restore_files()
            if restored:
                say(f"[evaluate] restored from the state store: {', '.join(restored)}")
    if not resume:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for path in (results_path, raw_path):
            if path.exists():
                os.replace(path, path.with_name(f"{path.name}.old-{stamp}"))
    for path in (results_path, raw_path):
        _drop_partial_tail(path)
    final = {run_id: row["status"] for run_id, row in read_results(results_path).items()}
    engine_name = str(getattr(engine, "name", type(engine).__name__))
    started_at = clock()
    event_base = {"engine": engine_name, "requests": len(items)}

    def event(**kw: Any) -> None:
        _atomic_json(out_dir / STATUS_NAME, {"time": _utc(), **event_base, **kw})

    environment: Dict[str, Any] = {"engine": engine_name, "model_source": dict(getattr(engine, "provenance", {}) or {}),
                                   "requests_file": Path(requests_path).name, "started_utc": _utc()}
    try:
        environment.update(engine.runtime())
    except Exception as exc:                                     # noqa: BLE001 - bookkeeping must not stop the run
        environment["runtime_error"] = f"{type(exc).__name__}: {exc}"
    _atomic_json(out_dir / ENVIRONMENT_NAME, environment)

    todo = [item for item in items if final.get(item.run_id) in (None, "error")]
    skipped = len(items) - len(todo)
    counts: collections.Counter = collections.Counter()
    new_rows = new_tokens = 0
    done_tokens = done_seconds = 0.0
    stop = "complete"
    say(f"[evaluate] {len(items)} requests: {skipped} already done, {len(todo)} to run")
    event(event="ready", todo=len(todo), skipped=skipped)
    position = 0
    while position < len(todo):
        size = chunk if limit is None else min(chunk, limit - new_rows)
        if size <= 0:
            stop = "limit"
            break
        piece = todo[position:position + size]
        if budget is not None:
            cost = (sum(i.tokens() for i in piece) / (done_tokens / done_seconds)) if done_seconds > 0 and done_tokens > 0 else 0.0
            if not budget.ok(cost):
                stop = "budget"
                say(f"[evaluate] stopping before the next chunk: it needs about {cost / 60:.1f} min and the budget has less")
                break
        t0 = clock()
        started = _utc()
        got = _call(engine, piece)
        try:
            engine.synchronize()
        except Exception:                                        # noqa: BLE001
            pass
        seconds = max(clock() - t0, 0.0)
        finished = _utc()
        wall_ms = seconds * 1000.0 / len(piece)
        rows, raws = [], []
        for item, answer in zip(piece, got):
            row, raw_line = _row_from(item, answer, engine_name, started, finished, wall_ms, len(piece))
            rows.append(_dumps(row))
            if raw_line:
                raws.append(raw_line)
            counts[row["status"]] += 1
            final[item.run_id] = row["status"]
            if row["status"] == "ok":
                new_tokens += int(((row.get("response") or {}).get("usage") or {}).get("input_tokens") or 0)
            done_tokens += item.tokens()
        _append(raw_path, raws)                                  # raw first: a result row always has its raw logits
        _append(results_path, rows)
        done_seconds += seconds
        new_rows += len(piece)
        position += len(piece)
        if state is not None:
            state.save()
        say(f"[evaluate] {new_rows}/{len(todo)} rows, {dict(counts)}")
        event(event="progress", done=new_rows, todo=len(todo), counts=dict(counts), elapsed_seconds=round(clock() - started_at, 1))
        if counts["error"] >= 5 and counts["ok"] == 0:
            event(event="failed", counts=dict(counts), error="five failed requests without a success")
            raise EvaluateError("five failed requests without a success; stopping instead of filling the file with errors "
                                f"(last error: {json.loads(rows[-1]).get('error')})")
    if stop == "complete" and limit is not None and new_rows >= limit and position < len(todo):
        stop = "limit"
    seconds_total = clock() - started_at
    if state is not None:
        state.push(files_now=True)
        if state.push_error:
            say(f"[evaluate] the state store did not take the files: {state.push_error}")
    files = {"results": str(results_path), "raw": str(raw_path), "status": str(out_dir / STATUS_NAME),
             "environment": str(out_dir / ENVIRONMENT_NAME)}
    if table and stop == "complete":
        try:
            files["table"] = write_table(out_dir, {i.run_id: i.gold for i in items if i.gold}, table_path)
        except EvaluateError as exc:
            say(f"[evaluate] no probability table: {exc}")
    event(event="complete" if stop == "complete" else "stopped", reason=stop, counts=dict(counts), total=len(final),
          elapsed_seconds=round(seconds_total, 1))
    return {"rows": new_rows, "total": len(final), "tokens": new_tokens, "seconds": seconds_total, "status": stop,
            "counts": dict(counts), "skipped": skipped, "out_dir": str(out_dir), "files": files}


# --------------------------------------------------------------------------- the probability table

TABLE_COLUMNS = ("run_id", "catalog_id", "group_id", "track", "question_id", "qtype", "status", "n_options", "option_keys", "codes",
                 "probs", "logits", "views", "temperature", "applied_temperature", "tokens", "gold", "error")


def _gold_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def table_rows(run_dir: Union[str, os.PathLike], golds: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Iterator[Dict[str, Any]]:
    """One dict per (request, question): option keys, probabilities after temperature and TTA, raw logits per view."""
    results = read_results(run_dir)
    raws = read_raw(run_dir)
    golds = golds or {}
    for run_id, row in results.items():
        base = {k: row.get(k) for k in ("run_id", "catalog_id", "group_id", "track", "status")}
        answers = (row.get("response") or {}).get("answers") if row.get("status") == "ok" else None
        if not answers:
            yield {**base, "question_id": None, "qtype": None, "n_options": 0, "option_keys": [], "codes": [], "probs": [], "logits": [],
                   "views": 0, "temperature": None, "applied_temperature": None, "tokens": 0, "gold": None, "error": row.get("error")}
            continue
        raw = raws.get(run_id) or {}
        tokens = int(((row.get("response") or {}).get("usage") or {}).get("input_tokens") or 0)
        for qid, answer in answers.items():
            if answer.get("type") == "noul":
                keys, probs = ["false", "true"], [1.0 - float(answer["noul"]), float(answer["noul"])]
            else:
                keys = list(answer["probabilities"])
                probs = [float(answer["probabilities"][k]) for k in keys]
            logits = (raw.get("logits") or {}).get(qid) or []
            yield {**base, "question_id": qid, "qtype": answer.get("type"), "n_options": len(keys), "option_keys": keys,
                   "codes": (raw.get("codes") or {}).get(qid), "probs": probs, "logits": [[float(x) for x in view] for view in logits], "views": int(raw.get("views") or len(logits)),
                   "temperature": raw.get("temperature"), "applied_temperature": (raw.get("applied_temperatures") or {}).get(qid),
                   "tokens": tokens, "gold": _gold_key((golds.get(run_id) or {}).get(qid)), "error": None}


def write_table(run_dir: Union[str, os.PathLike], golds: Optional[Mapping[str, Mapping[str, Any]]] = None,
                path: Union[str, os.PathLike, None] = None) -> str:
    """Write the probability table as parquet (needs pyarrow).  The JSONL files stay the source of truth."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvaluateError("writing the probability table needs pyarrow (pip install pyarrow)") from exc
    out = Path(path) if path else Path(run_dir) / TABLE_NAME
    schema = pa.schema([
        ("run_id", pa.string()), ("catalog_id", pa.int64()), ("group_id", pa.string()), ("track", pa.string()),
        ("question_id", pa.string()), ("qtype", pa.string()), ("status", pa.string()), ("n_options", pa.int32()),
        ("option_keys", pa.list_(pa.string())), ("codes", pa.list_(pa.string())), ("probs", pa.list_(pa.float64())),
        ("logits", pa.list_(pa.list_(pa.float64()))),
        ("views", pa.int32()), ("temperature", pa.float64()), ("applied_temperature", pa.float64()), ("tokens", pa.int64()),
        ("gold", pa.string()), ("error", pa.string()),
    ])
    rows = list(table_rows(run_dir, golds))
    tmp = out.with_name(out.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(tmp))
    os.replace(tmp, out)
    return str(out)


def _table_file(path: Union[str, os.PathLike]) -> Path:
    p = Path(path)
    p = p / TABLE_NAME if p.is_dir() else p
    if not p.exists():
        raise EvaluateError(f"{p}: no probability table here; run_requests writes it when a run completes")
    return p


def _table_batches(path: Union[str, os.PathLike], columns: Sequence[str]) -> Iterator[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvaluateError("reading the probability table needs pyarrow (pip install pyarrow)") from exc
    for batch in pq.ParquetFile(str(_table_file(path))).iter_batches(batch_size=1024, columns=list(columns)):
        yield from batch.to_pylist()


def table_items(path: Union[str, os.PathLike]) -> List[calibrate.Item]:
    """`calibrate.Item`s from a probability table (the file, or the run directory that holds it).

    Rows that are not `ok`, have no gold label, or whose gold is not one of the option keys are left out.
    """
    items: List[calibrate.Item] = []
    for row in _table_batches(path, ("run_id", "question_id", "qtype", "status", "option_keys", "logits", "gold")):
        keys, gold, views = row.get("option_keys") or [], row.get("gold"), row.get("logits") or []
        if row.get("status") != "ok" or row.get("question_id") is None or gold is None or gold not in keys:
            continue
        if not views or any(len(view) != len(keys) for view in views):
            continue
        items.append(calibrate.Item(row["run_id"], row["question_id"], str(row["qtype"]), [[float(x) for x in v] for v in views], keys.index(gold)))
    return items


def table_temperature(path: Union[str, os.PathLike]) -> Optional[float]:
    """The temperature the model itself ships with (the most common value in the table), or None when nothing was recorded."""
    counts: collections.Counter = collections.Counter()
    for row in _table_batches(path, ("status", "temperature")):
        if row.get("status") == "ok" and row.get("temperature") is not None:
            counts[float(row["temperature"])] += 1
    return counts.most_common(1)[0][0] if counts else None


# --------------------------------------------------------------------------- throughput and the sample size

def _spread(ordered: Sequence[Any], n: int) -> List[Any]:
    if n >= len(ordered):
        return list(ordered)
    if n == 1:
        return [ordered[len(ordered) // 2]]
    return [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]


def measure_throughput(engine: Any, items: Sequence[Item], *, n: int = 8, warmup: int = 1,
                       clock: Callable[[], float] = time.perf_counter) -> Dict[str, Any]:
    """Proxy tokens per second of one batch of `n` requests spread over the length range, after `warmup` small requests.

    The tokens are the rows' own `proxy_tokens`, the unit `sample.fit` budgets in, so the figure already includes the
    prompt overhead and the number of TTA views.  Nothing is written.
    """
    if not items:
        raise EvaluateError("no requests to measure the throughput on")
    ordered = sorted(items, key=lambda i: (i.tokens(), i.run_id))
    picks = _spread(ordered, max(1, n))
    for item in ordered[:max(0, warmup)]:
        _call(engine, [item])
    _sync(engine)
    t0 = clock()
    got = _call(engine, picks)
    _sync(engine)
    seconds = max(clock() - t0, 1e-9)
    failed = [g for g in got if isinstance(g, BaseException)]
    if len(failed) == len(got):
        raise EvaluateError(f"every probe request failed; first error: {type(failed[0]).__name__}: {failed[0]}")
    tokens = sum(i.tokens() for i in picks)
    return {"n": len(picks), "seconds": seconds, "proxy_tokens": tokens, "tokens_per_second": tokens / seconds,
            "requests_per_second": len(picks) / seconds, "run_ids": [i.run_id for i in picks], "failed": len(failed)}


def _sync(engine: Any) -> None:
    try:
        engine.synchronize()
    except Exception:                                            # noqa: BLE001
        pass


def fit_to_budget(rows: Iterable[Mapping[str, Any]], *, tokens_per_second: float, seconds: float, seed: int,
                  minimum: int = 1, safety: float = 0.85) -> Dict[str, Any]:
    """The largest groups-per-benchmark whose proxy tokens fit in `seconds` at the measured throughput, with a safety factor."""
    if not tokens_per_second > 0 or not math.isfinite(tokens_per_second):
        raise EvaluateError("the measured throughput must be positive")
    budget_tokens = tokens_per_second * max(0.0, seconds) * safety
    fit = sample_mod.fit(rows, seed=seed, token_budget=budget_tokens, minimum=minimum)
    where = (f"budget {budget_tokens:,.0f} proxy tokens = {tokens_per_second:.1f} tok/s x {seconds / 3600:.2f} h x {safety}")
    if not fit["fits"]:
        message = (f"even {fit['per_benchmark']} groups per benchmark does not fit: it needs {fit['tokens']:,} proxy tokens, {where}")
    elif fit["per_benchmark"] < fit["current"]:
        message = f"sample shrunk from {fit['current']} to {fit['per_benchmark']} groups per benchmark ({fit['tokens']:,} proxy tokens; {where})"
    else:
        message = f"sample kept at {fit['per_benchmark']} groups per benchmark ({fit['tokens']:,} proxy tokens; {where})"
    return {**fit, "token_budget": budget_tokens, "tokens_per_second": tokens_per_second, "seconds": seconds, "safety": safety,
            "message": message}


# --------------------------------------------------------------------------- scoring through the kit

def _kit() -> Dict[str, Any]:
    try:
        from decision_index import editions
        from decision_index.scoring import added, index02
        from decision_index.scoring.index import score_panel
        from decision_index.scoring.report import benchmark_summary, load_results
    except ImportError as exc:
        raise EvaluateError("scoring needs the Decision Index kit: pip install "
                            "git+https://github.com/apolinario/decision-index (commit in decide/config.yaml, kit.commit)") from exc
    return {"editions": editions, "added": added, "index02": index02, "score_panel": score_panel,
            "benchmark_summary": benchmark_summary, "load_results": load_results}


def _restrict(spec: Mapping[str, Any], present: Iterable[int]) -> Dict[str, Any]:
    """The index spec with only the benchmarks (and areas) that are in the sample; weights renormalise over what is left."""
    keep = set(present)
    mod = copy.deepcopy(dict(spec))
    areas = []
    for area in mod["areas"]:
        area["benchmarks"] = [n for n in area["benchmarks"] if n in keep]
        if area["benchmarks"]:
            areas.append(area)
    if not areas:
        raise EvaluateError("none of the sampled benchmarks is part of the index")
    mod["areas"] = areas
    sizing = mod.get("area_sizing")
    if sizing:
        names = {a["id"] for a in areas}
        sizing["fixed"] = {k: v for k, v in sizing["fixed"].items() if k in names}
    mod["not_in_index"] = []
    return mod


def _score(results: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], edition: str, engine: str) -> Dict[str, Any]:
    kit = _kit()
    index02 = kit["index02"]
    spec = index02.spec(edition)
    suite = sample_mod.SampleSuite.from_rows(rows, kit["editions"].get(edition))
    added_ids = {int(n) for n in spec["added"]}
    base, extra = [], collections.defaultdict(list)
    for row in rows:
        n = row["_evaluation"]["catalog_id"]
        (extra[n] if n in added_ids else base).append(row)
    metrics = {int(n): (m["name"], m["key"]) for n, m in spec.get("metrics", {}).items()}
    summary = kit["benchmark_summary"](suite, results, engine, None, rows=base, metrics=metrics)
    added_reports = {n: kit["added"].report(n, rs, results) for n, rs in sorted(extra.items())}
    scored = kit["score_panel"](suite, results)
    native = {b["catalog_id"]: b for b in summary["benchmarks"]}
    native.update(added_reports)
    present = {row["_evaluation"]["catalog_id"] for row in rows}
    indexed = {n for area in spec["areas"] for n in area["benchmarks"]}
    values = {n: index02.benchmark_value(n, spec, scored.get(n), native.get(n)) for n in sorted(indexed & present)}
    mod = _restrict(spec, values)
    scores, areas = index02.aggregate(values, mod)
    per_benchmark: Dict[str, Any] = {}
    for n, value in values.items():
        entry = native.get(n) or {}
        per_benchmark[str(n)] = {**{k: value[k] for k in ("raw", "skill", "coverage", "random", "rule")},
                                 **{k: entry.get(k) for k in ("dataset", "metric", "score", "requests", "answered", "unsupported",
                                                              "errors", "pending") if k in entry}}
    status_counts = collections.Counter(results[row["_evaluation"]["run_id"]]["status"] if row["_evaluation"]["run_id"] in results
                                        else "pending" for row in rows)
    return {
        "public": scores["balanced_skill"], "raw": scores["balanced_raw"], "scores": scores, "edition": edition,
        "areas": {a["id"]: {"skill": 100.0 * a["skill"], "raw": 100.0 * a["raw"], "coverage": a["coverage"], "n": a["n"]} for a in areas},
        "per_benchmark": per_benchmark, "benchmarks_present": sorted(values),
        "benchmarks_missing": sorted(indexed - set(values)), "rows": len(rows), "counts": dict(status_counts),
        "partial": True, "note": NOTE,
    }


def score_results(results_path: Union[str, os.PathLike], rows: Sequence[Mapping[str, Any]], *, edition: str = "0.3",
                  engine: str = "simplicio-decide") -> Dict[str, Any]:
    """The sample index of a results file over `rows` (suite rows with gold), by the kit's own scorer."""
    return _score(_kit()["load_results"](str(results_path)), rows, edition, engine)


def _sample_dir(ref: Union[str, os.PathLike]) -> Path:
    p = Path(ref)
    return p.parent if p.is_file() or p.name == sample_mod.MANIFEST_NAME else p


def score_sample(results_path: Union[str, os.PathLike], sample_ref: Union[str, os.PathLike], *, edition: Optional[str] = None,
                 engine: str = "simplicio-decide") -> Dict[str, Any]:
    """`score_results` over a sample written by `sample.write` (the directory, or its manifest file)."""
    rows, manifest = sample_mod.load(_sample_dir(sample_ref))
    out = score_results(results_path, rows, edition=edition or manifest.get("edition") or "0.3", engine=engine)
    out["sample"] = manifest
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvaluateError("no replicates")
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def compare_runs(results_a: Union[str, os.PathLike], results_b: Union[str, os.PathLike], rows: Sequence[Mapping[str, Any]], *,
                 edition: str = "0.3", replicates: int = 200, seed: int = 0, level: float = 0.95) -> Dict[str, Any]:
    """Paired difference of the sample index, A minus B, with a percentile bootstrap over groups (cases) within each benchmark."""
    kit = _kit()
    res_a, res_b = kit["load_results"](str(results_a)), kit["load_results"](str(results_b))
    a = _score(res_a, rows, edition, "a")["public"]
    b = _score(res_b, rows, edition, "b")["public"]
    cases: Dict[Any, Dict[Any, List[Mapping[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        e = row["_evaluation"]
        cases[e["catalog_id"]][e["group_id"]].append(row)
    rng = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        sampled: List[Mapping[str, Any]] = []
        for benchmark in sorted(cases):
            groups = list(cases[benchmark].values())
            for _g in groups:
                sampled.extend(groups[rng.randrange(len(groups))])
        deltas.append(_score(res_a, sampled, edition, "a")["public"] - _score(res_b, sampled, edition, "b")["public"])
    tail = (1.0 - level) / 2.0
    return {"delta": a - b, "a": a, "b": b, "low": _percentile(deltas, tail) if deltas else None,
            "high": _percentile(deltas, 1.0 - tail) if deltas else None, "replicates": replicates, "seed": seed, "level": level}


# --------------------------------------------------------------------------- items for calibration

def _gold_index(question: Mapping[str, Any], expected: Any) -> Optional[int]:
    if question.get("type") == "noul":
        if expected is True or expected == "true":
            return 1
        if expected is False or expected == "false":
            return 0
        return None
    keys = list(question.get("criteria") or {})
    return keys.index(expected) if isinstance(expected, str) and expected in keys else None


def calibration_items(run_dir: Union[str, os.PathLike], rows: Sequence[Mapping[str, Any]]) -> List[calibrate.Item]:
    """`calibrate.Item`s from the raw logits of a run and the gold labels of the sample rows (unlabelled questions are left out)."""
    by_id = {row["_evaluation"]["run_id"]: row for row in rows}
    items: List[calibrate.Item] = []
    for run_id, raw in read_raw(run_dir).items():
        row = by_id.get(run_id)
        if row is None:
            continue
        for qid, views in (raw.get("logits") or {}).items():
            question = (row.get("questions") or {}).get(qid)
            if question is None or qid not in (row.get("expected") or {}):
                continue
            gold = _gold_index(question, row["expected"][qid])
            width = 2 if question.get("type") == "noul" else len(question.get("criteria") or {})
            if gold is None or not views or any(len(v) != width for v in views):
                continue
            items.append(calibrate.Item(run_id, qid, str(question["type"]), [list(map(float, v)) for v in views], gold))
    return items


# --------------------------------------------------------------------------- one sample, end to end

def evaluate_sample(engine: Any, sample_dir: Union[str, os.PathLike], out_dir: Union[str, os.PathLike], *, budget: Any = None,
                    per_benchmark: Optional[int] = None, probe: int = 8, safety: float = 0.85, minimum: int = 1, chunk: int = 8,
                    state: Any = None, resume: bool = True, log: Optional[Callable[[str], None]] = print, score: bool = True,
                    name: Optional[str] = None, clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """Measure the throughput, shrink the sample to the budget (logged), run it, score it, write `metrics.json`.

    The chosen size is stored in `plan.json`, so a resumed run keeps the same rows.  `per_benchmark` forces a size.
    """
    say = log or (lambda text: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, manifest = sample_mod.load(_sample_dir(sample_dir))
    seed = int(manifest["seed"])
    plan_path = out / PLAN_NAME
    plan: Dict[str, Any]
    if resume and plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        say(f"[evaluate] resuming with the sample size chosen before: {plan['per_benchmark']} groups per benchmark")
    else:
        plan = {"per_benchmark": per_benchmark or int(manifest["per_benchmark"]), "forced": per_benchmark is not None, "message": "sample used as written"}
        if per_benchmark is None and budget is not None:
            probe_out = measure_throughput(engine, items_from_rows(rows), n=probe)
            fit = fit_to_budget(rows, tokens_per_second=probe_out["tokens_per_second"], seconds=budget.available(), seed=seed,
                                minimum=minimum, safety=safety)
            plan = {"per_benchmark": fit["per_benchmark"], "forced": False, "message": fit["message"], "fits": fit["fits"],
                    "tokens_per_second": probe_out["tokens_per_second"], "token_budget": fit["token_budget"], "proxy_tokens": fit["tokens"]}
        say(f"[evaluate] {plan['message']}")
        _atomic_json(plan_path, plan)
    selection = sample_mod.select(rows, per_benchmark=int(plan["per_benchmark"]), seed=seed,
                                  max_proxy_tokens=manifest.get("max_proxy_tokens"), edition=manifest.get("edition"))
    subset_dir = out / SUBSET_DIR
    sample_mod.write(selection, subset_dir)
    run = run_requests(engine, subset_dir / sample_mod.ROWS_NAME, out / "run", budget=budget, resume=resume, chunk=chunk, state=state,
                       state_prefix=f"runs/{name or out.name}", log=log, clock=clock)
    metrics: Dict[str, Any] = {"name": name or out.name, "plan": plan, "run": {k: run[k] for k in ("rows", "total", "tokens", "seconds", "status", "counts")},
                               "sample": selection.manifest["totals"], "edition": manifest.get("edition")}
    if score and run["status"] == "complete":
        metrics["scores"] = score_results(out / "run" / RESULTS_NAME, selection.rows, edition=manifest.get("edition") or "0.3",
                                          engine=name or out.name)
    _atomic_json(out / METRICS_NAME, metrics)
    return metrics
