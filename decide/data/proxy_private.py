"""The private home-made proxy ("privado caseiro"): held-out new-domain tasks, scoring and the G1 check.

* ``build_proxy`` builds the proxy only from sources marked ``role: heldout`` in sources.yaml; those never train.
  It refuses to run if the corpus manifest shares a source with the proxy, and (with ``corpus``) if any corpus
  request leaks a proxy request (``LeakError``).
* ``score_rows`` / ``score_proxy`` score a per-row probabilities file: chance-corrected accuracy per source,
  averaged over sources, plus NLL and Brier.  It is *not* the Jev metric (each Jev benchmark has its own).
* ``validate_against_board`` is gate G1: Spearman between the proxy scores of >= 6 open models and their Full
  score on the board (``index.json``, field ``models[].v03.v``).  Also reported against ``new_domains``,
  ``same_skills`` and ``pub``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from . import decontaminate as D
from . import schema as S
from . import sources as SRC
from .ingest import convert_source

DEFAULT_PER_SOURCE = 300
DEFAULT_MIN_MODELS = 6
DEFAULT_MIN_RHO = 0.8
EXACT_PERMUTATION_MAX_N = 8
MONTE_CARLO_DRAWS = 20_000


class LeakError(RuntimeError):
    """The proxy and the training corpus share a source or a request."""


def assert_disjoint(corpus_sources: Iterable[str], proxy_sources: Iterable[str]) -> None:
    shared = sorted(set(corpus_sources) & set(proxy_sources))
    if shared:
        raise LeakError(f"sources in both the corpus and the proxy: {shared}")


# --------------------------------------------------------------------------- statistics

def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        raise ValueError("a constant series has no rank correlation")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("at least two points are needed")
    return _pearson(_ranks(x), _ranks(y))


def permutation_p_value(x: Sequence[float], y: Sequence[float], seed: int = 0) -> float:
    """One-sided p-value of observing a Spearman >= the observed one under random pairing (exact for n <= 8)."""
    rx, ry = _ranks(x), _ranks(y)
    observed = _pearson(rx, ry)
    n = len(rx)
    if n <= EXACT_PERMUTATION_MAX_N:
        total = hits = 0
        for perm in itertools.permutations(ry):
            total += 1
            hits += _pearson(rx, list(perm)) >= observed - 1e-12
        return hits / total
    rng = random.Random(seed)
    hits = 0
    shuffled = list(ry)
    for _ in range(MONTE_CARLO_DRAWS):
        rng.shuffle(shuffled)
        hits += _pearson(rx, shuffled) >= observed - 1e-12
    return (hits + 1) / (MONTE_CARLO_DRAWS + 1)


# --------------------------------------------------------------------------- the board (gate G1)

def load_board(board: Union[str, Path, Mapping[str, Any]]) -> Mapping[str, Any]:
    return board if isinstance(board, Mapping) else json.loads(Path(board).read_text(encoding="utf-8"))


def board_models(board: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """engine -> the model's ``v03`` block (Full ``v``, ``same_skills``, ``new_domains``, ``pub``).

    The engine is the canonical id of a model (unique on the real board: 208 models, 208 engines)."""
    out: Dict[str, Dict[str, Any]] = {}
    for m in board.get("models", []):
        v03 = m.get("v03")
        if isinstance(v03, Mapping) and isinstance(v03.get("v"), (int, float)):
            out[str(m.get("engine"))] = dict(v03)
    return out


def board_aliases(board: Mapping[str, Any]) -> Dict[str, List[str]]:
    """display name -> the engines that carry it.  A name carried by two models identifies neither."""
    out: Dict[str, List[str]] = {}
    for m in board.get("models", []):
        v03 = m.get("v03")
        if isinstance(v03, Mapping) and isinstance(v03.get("v"), (int, float)):
            engines = out.setdefault(str(m.get("name")), [])
            if str(m.get("engine")) not in engines:
                engines.append(str(m.get("engine")))
    return out


def select_models(board: Mapping[str, Any], *, max_params_b: Optional[float] = None, kinds: Optional[Sequence[str]] = None,
                  limit: int = 12) -> List[Dict[str, Any]]:
    """Candidate models to run on the proxy: spread over the Full range, optionally small and of given kinds."""
    rows = []
    for m in board.get("models", []):
        v03, meta = m.get("v03") or {}, m.get("meta") or {}
        params = meta.get("served_params")
        if not isinstance(v03.get("v"), (int, float)):
            continue
        if max_params_b is not None and not (isinstance(params, (int, float)) and params <= max_params_b * 1e9):
            continue
        if kinds and meta.get("kind") not in kinds:
            continue
        rows.append({"engine": m.get("engine"), "name": m.get("name"), "full": v03["v"], "params": params, "kind": meta.get("kind"),
                     "base_model": meta.get("base_model")})
    rows.sort(key=lambda r: r["full"])
    if len(rows) <= limit:
        return rows
    picks = sorted({round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)})
    return [rows[i] for i in picks]


def validate_against_board(proxy_scores: Mapping[str, float], board: Union[str, Path, Mapping[str, Any]], *,
                           min_models: int = DEFAULT_MIN_MODELS, min_rho: float = DEFAULT_MIN_RHO, seed: int = 0) -> Dict[str, Any]:
    """Gate G1: Spearman(proxy, board Full) over the models present in both.

    A key of ``proxy_scores`` is a board engine, or a display name that belongs to exactly one model (the engine wins
    when a name equals another model's engine).  Every key is resolved to the engine, which is what is counted, so a
    model never counts twice.  Two keys for the same model are refused (``passed`` false, ``duplicates``): the n of the
    "at least 6 open models" bar and the correlation would be inflated by repeats."""
    loaded = load_board(board)
    known, aliases = board_models(loaded), board_aliases(loaded)
    keys_of: Dict[str, List[str]] = {}
    unmatched: List[str] = []
    ambiguous: List[str] = []
    for key in sorted(proxy_scores):
        if key in known:
            keys_of.setdefault(key, []).append(key)
        elif len(aliases.get(key, [])) == 1:
            keys_of.setdefault(aliases[key][0], []).append(key)
        elif key in aliases:
            ambiguous.append(key)
        else:
            unmatched.append(key)
    matched = sorted(keys_of)
    duplicates = {engine: keys for engine, keys in sorted(keys_of.items()) if len(keys) > 1}
    scores = {engine: float(proxy_scores[keys[0]]) for engine, keys in keys_of.items()}
    result: Dict[str, Any] = {"n": len(matched), "models": matched, "unmatched": unmatched, "ambiguous": ambiguous, "duplicates": duplicates,
                              "min_models": min_models, "min_rho": min_rho,
                              "rho_full": None, "rho_new_domains": None, "rho_same_skills": None, "rho_pub": None, "p_value": None,
                              "passed": False, "reason": ""}
    if duplicates:
        result["reason"] = (f"{len(duplicates)} board models are listed under more than one key ({'; '.join(f'{e}: {k}' for e, k in duplicates.items())}); "
                            "give every model once, by engine")
        return result
    if len(matched) < 2:
        result["reason"] = f"need at least {min_models} models, matched {len(matched)}"
        return result
    xs = [scores[m] for m in matched]
    for key, field in (("rho_full", "v"), ("rho_new_domains", "new_domains"), ("rho_same_skills", "same_skills"), ("rho_pub", "pub")):
        ys = [known[m].get(field) for m in matched]
        if all(isinstance(v, (int, float)) for v in ys):
            try:
                result[key] = spearman(xs, [float(v) for v in ys])
            except ValueError:
                result[key] = None
    if result["rho_full"] is not None:
        result["p_value"] = permutation_p_value(xs, [float(known[m]["v"]) for m in matched], seed)
    if len(matched) < min_models:
        result["reason"] = f"need at least {min_models} models, matched {len(matched)}"
    elif result["rho_full"] is None:
        result["reason"] = "the proxy scores or the board scores are constant"
    elif result["rho_full"] < min_rho:
        result["reason"] = f"Spearman {result['rho_full']:.3f} < {min_rho}"
    else:
        result["passed"] = True
        result["reason"] = f"Spearman {result['rho_full']:.3f} >= {min_rho} over {len(matched)} models"
    return result


# --------------------------------------------------------------------------- scoring

_ID_ALIASES = ("example_id", "request_id", "run_id", "id")


def read_results(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Per-row probabilities (parquet or JSONL): ``example_id`` (or request_id/run_id/id), ``question_id``, ``codes``, ``probs``."""
    p = Path(path)
    if p.is_dir() or p.suffix == ".parquet":
        _, pq = S._arrow()
        for f in S.corpus_files(p):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=S.BATCH_ROWS):
                yield from batch.to_pylist()
        return
    import gzip

    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[call-overload]
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _row_id(row: Mapping[str, Any]) -> Optional[str]:
    for name in _ID_ALIASES:
        if row.get(name) is not None:
            return str(row[name])
    return None


def score_rows(requests: Iterable[Tuple[str, Mapping[str, Any], Mapping[str, Any], str]], results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score ``results`` against ``requests`` = (example_id, request, gold, source) tuples."""
    table: Dict[Tuple[str, str], List[float]] = {}
    for r in results:
        rid, qid = _row_id(r), r.get("question_id")
        probs = r.get("probs", r.get("probabilities"))
        if rid is not None and qid is not None and probs is not None:
            table[(rid, str(qid))] = [float(p) for p in probs]
    per: Dict[str, Dict[str, float]] = {}
    nll = brier = 0.0
    n = missing = bad = 0
    for eid, request, gold, source in requests:
        for qid, key in gold.items():
            if key is None:
                continue
            question = request["questions"][qid]
            keys = S.option_keys(question)
            probs = table.get((eid, qid))
            if probs is None:
                missing += 1
                continue
            if len(probs) != len(keys) or sum(probs) <= 0 or any(p < 0 or not math.isfinite(p) for p in probs):
                bad += 1
                continue
            total = sum(probs)
            p = [x / total for x in probs]
            g = keys.index(key)
            agg = per.setdefault(source, {"n": 0.0, "correct": 0.0, "chance": 0.0})
            agg["n"] += 1
            agg["correct"] += 1.0 if max(range(len(p)), key=p.__getitem__) == g else 0.0
            agg["chance"] += 1.0 / len(keys)
            nll += -math.log(max(p[g], 1e-12))
            brier += sum((pi - (1.0 if i == g else 0.0)) ** 2 for i, pi in enumerate(p))
            n += 1
    per_source: Dict[str, Dict[str, float]] = {}
    for source, agg in sorted(per.items()):
        acc, chance = agg["correct"] / agg["n"], agg["chance"] / agg["n"]
        per_source[source] = {"n": int(agg["n"]), "accuracy": acc, "chance": chance, "skill": (acc - chance) / (1 - chance) if chance < 1 else 0.0}
    skills = [v["skill"] for v in per_source.values()]
    return {"n": n, "missing": missing, "bad": bad, "proxy": sum(skills) / len(skills) if skills else None,
            "nll": nll / n if n else None, "brier": brier / n if n else None, "per_source": per_source}


def score_proxy(results_path: Union[str, Path], proxy_path: Union[str, Path]) -> Dict[str, Any]:
    requests = ((row["example_id"], *S.parse_row(row), row["source"]) for row in S.read_corpus(proxy_path))
    return score_rows(((eid, req, gold, src) for eid, (req, gold), src in ((a, (b, c), d) for a, b, c, d in requests)), read_results(results_path))


# --------------------------------------------------------------------------- building

def build_proxy(out_dir: Path, *, sources: Mapping[str, SRC.Source], reader: Any, corpus_manifest: Optional[Union[str, Path, Mapping[str, Any]]] = None,
                corpus: Optional[Union[str, Path]] = None, per_source: int = DEFAULT_PER_SOURCE, seed: int = 20261021,
                include_review: bool = False, extra_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Write ``proxy.parquet`` (corpus schema, split ``dev``) from the held-out sources, plus ``manifest.json``."""
    SRC.check_policy(sources)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chosen = [s for _, s in sorted(sources.items()) if SRC.usable(s, include_review=include_review, role="heldout")]
    if not chosen and not extra_rows:
        raise ValueError("no held-out sources: mark some sources 'role: heldout' in sources.yaml")
    if corpus_manifest is not None:
        manifest = corpus_manifest if isinstance(corpus_manifest, Mapping) else json.loads(Path(corpus_manifest).read_text(encoding="utf-8"))
        assert_disjoint(manifest.get("sources", {}), [s.key for s in chosen])
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    for source in chosen:
        got, st = convert_source(source, reader.open(source), seed=seed, cap=per_source)
        rows += got
        stats[source.key] = {"rows": len(got), "license": source.license, "stats": st}
    for r in extra_rows or []:
        rows.append(dict(r))
        stats.setdefault(r["source"], {"rows": 0, "license": r["license"], "stats": {}})["rows"] += 1
    built = [S.corpus_row(example_id=r["example_id"], source=r["source"], license=r["license"], split="dev", lang=r["lang"],
                          request=r["request"], gold=r["gold"]) for r in rows]
    info = S.write_corpus_part(out / "proxy.parquet", built)
    leaks: Dict[str, Any] = {"checked": False}
    if corpus is not None:
        index = D.ReferenceIndex()
        for r in built:
            index.add(S.parse_row(r)[0], r["example_id"])
        index.freeze()
        hits = []
        for row in S.read_corpus(corpus):
            m = index.check(S.parse_row(row)[0])
            if m:
                hits.append({"example_id": row["example_id"], "reason": m.reason})
        leaks = {"checked": True, "leaks": len(hits), "examples": hits[:20]}
        if hits:
            raise LeakError(f"{len(hits)} corpus examples overlap the proxy, e.g. {hits[:3]}")
    result = {"schema_version": S.SCHEMA_VERSION, "kind": "proxy", "seed": seed, "per_source": per_source, "rows": len(built),
              "file": info, "sources": stats, "corpus_leak_check": leaks}
    S.write_manifest(out / "manifest.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build, score and validate the private home-made proxy.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build proxy.parquet from the held-out sources")
    b.add_argument("--out", required=True)
    b.add_argument("--reader", default="hub")
    b.add_argument("--sources", type=Path, default=None)
    b.add_argument("--per-source", type=int, default=DEFAULT_PER_SOURCE)
    b.add_argument("--corpus-manifest", default=None, help="corpus manifest.json (sources must not overlap)")
    b.add_argument("--corpus", default=None, help="corpus directory to scan for leaks of proxy requests")
    b.add_argument("--include-review", action="store_true")
    s = sub.add_parser("score", help="score a per-row probabilities file on the proxy")
    s.add_argument("--results", required=True)
    s.add_argument("--proxy", required=True)
    v = sub.add_parser("validate", help="gate G1: Spearman between proxy scores and the board Full score")
    v.add_argument("--scores", required=True, help="JSON {model: proxy score}")
    v.add_argument("--board", required=True, help="board index.json")
    v.add_argument("--min-models", type=int, default=DEFAULT_MIN_MODELS)
    v.add_argument("--min-rho", type=float, default=DEFAULT_MIN_RHO)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        from .build_corpus import make_reader

        res = build_proxy(Path(args.out), sources=SRC.load_sources(args.sources), reader=make_reader(args.reader), corpus_manifest=args.corpus_manifest,
                          corpus=args.corpus, per_source=args.per_source, include_review=args.include_review)
        print(json.dumps({"rows": res["rows"], "sources": {k: v["rows"] for k, v in res["sources"].items()}}, indent=2))
    elif args.cmd == "score":
        print(json.dumps(score_proxy(args.results, args.proxy), indent=2))
    else:
        res = validate_against_board(json.loads(Path(args.scores).read_text(encoding="utf-8")), args.board, min_models=args.min_models, min_rho=args.min_rho)
        print(json.dumps(res, indent=2))
        return 0 if res["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
