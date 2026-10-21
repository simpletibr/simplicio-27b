"""Build the training corpus: licensed sources -> adapters -> parquet parts + manifest.

Sources come from sources.yaml (deny by default; ``role: train`` and ``allow`` unless ``--include-review``), from
the Simplicio-Loop traces (``--loop``) and from the optional generator (off by default, see generate.py).  Each
source is written as its own parquet parts and marked done, so a rerun skips what is finished.

``--out`` always holds exactly the corpus of the last finished run: ``manifest.json`` is removed when a run starts
and written when it ends, it lists the parts (with sha256) and every reader goes through it; parts of sources that are
no longer selected (``--only``, ``--include-review`` off) and shards a smaller cap no longer writes are deleted.
Use one ``--out`` per corpus.

    python -m decide.data.build_corpus --out work/corpus --reader hub
    python -m decide.data.build_corpus --out work/corpus --reader local:work/raw --only clinc_oos,gsm8k
    python -m decide.data.build_corpus --inspect clinc_oos --reader hub      # first rows and their conversion
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from . import schema as S
from . import sources as SRC
from .ingest import convert_source
from .readers import HubReader, LocalReader

DEFAULT_SEED = 20261021
DEFAULT_DEV_PERCENT = 1.0
DEFAULT_SHARD_ROWS = 50_000


def split_of(example_id: str, dev_percent: float) -> str:
    """``dev`` for ~dev_percent % of the examples, chosen by the hash of the id (stable under reordering)."""
    return "dev" if int(S.digest("split:" + example_id)[:8], 16) % 10_000 < dev_percent * 100 else "train"


def _write_rows(out: Path, name: str, rows: Sequence[Mapping[str, Any]], dev_percent: float, shard_rows: int) -> List[Dict[str, Any]]:
    parts = []
    for k, chunk in enumerate(S.iter_chunks(rows, shard_rows)):
        built = [S.corpus_row(example_id=r["example_id"], source=r["source"], license=r["license"],
                              split=split_of(r["example_id"], dev_percent), lang=r["lang"], request=r["request"], gold=r["gold"])
                 for r in chunk]
        parts.append(S.write_corpus_part(out / f"{name}-{k:04d}.parquet", built))
    return parts


def _config_hash(source: Optional[SRC.Source], extra: Mapping[str, Any]) -> str:
    material = {"source": None if source is None else {k: getattr(source, k) for k in source.__dataclass_fields__}, **extra}
    return S.digest(S.canonical_json(material))


def _done_path(out: Path, name: str) -> Path:
    return out / f"{name}.done.json"


_PART_NAME = re.compile(r"-\d{4}\.parquet$")


def _prune_stale(out: Path, keep_parts: Iterable[str], keep_markers: Iterable[str]) -> List[str]:
    """Delete what an earlier run left in ``out`` and this run does not list: parts of sources that left the build
    (``--include-review`` turned off, a source turned from allow to deny), shards a smaller cap no longer writes,
    their ``.done.json`` markers and half-written ``.tmp`` files.  Only files named like this builder's own
    outputs are touched; any other file in the directory is left alone (and, not being in the manifest, never read)."""
    parts, markers = set(keep_parts), set(keep_markers)
    removed: List[str] = []
    for f in sorted(out.iterdir()):
        name = f.name
        stale = f.is_file() and ((_PART_NAME.search(name) and name not in parts)
                                 or (name.endswith(".done.json") and name not in markers)
                                 or name.endswith((".parquet.tmp", ".json.tmp")))
        if stale:
            f.unlink()
            removed.append(name)
    return removed


def _already_done(out: Path, name: str, cfg: str, resume: bool) -> Optional[Dict[str, Any]]:
    path = _done_path(out, name)
    if not (resume and path.exists()):
        return None
    done = json.loads(path.read_text(encoding="utf-8"))
    ok = done.get("config_hash") == cfg and all((out / p["name"]).exists() and S.sha256_file(out / p["name"]) == p["sha256"] for p in done["parts"])
    return done if ok else None


def build_corpus(out_dir: Path, *, sources: Mapping[str, SRC.Source], reader: Any, seed: int = DEFAULT_SEED,
                 dev_percent: float = DEFAULT_DEV_PERCENT, include_review: bool = False, only: Optional[Iterable[str]] = None,
                 caps_scale: float = 1.0, shard_rows: int = DEFAULT_SHARD_ROWS, extra_rows: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
                 resume: bool = True) -> Dict[str, Any]:
    """Write the corpus parts of every usable source plus ``extra_rows`` (name -> rows from loop/generator)."""
    SRC.check_policy(sources)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # the manifest of an earlier run describes files this run is about to replace or delete: a directory without a
    # manifest is not finished, and readers (which go through the manifest) refuse it until this run writes a new one
    (out / S.MANIFEST_NAME).unlink(missing_ok=True)
    wanted = set(only) if only else None
    chosen = [s for k, s in sorted(sources.items()) if SRC.usable(s, include_review=include_review) and (wanted is None or k in wanted)]
    if wanted and not wanted <= set(sources):
        raise KeyError(f"unknown sources: {sorted(wanted - set(sources))}")
    report: Dict[str, Dict[str, Any]] = {}
    resumed: List[str] = []
    for source in chosen:
        cap = None if source.cap is None else max(1, int(source.cap * caps_scale))
        cfg = _config_hash(source, {"seed": seed, "cap": cap, "dev_percent": dev_percent, "shard_rows": shard_rows})
        done = _already_done(out, source.key, cfg, resume)
        if done:
            resumed.append(source.key)
            report[source.key] = done
            continue
        rows, stats = convert_source(source, reader.open(source), seed=seed, cap=cap)
        parts = _write_rows(out, source.key, rows, dev_percent, shard_rows)
        report[source.key] = {"config_hash": cfg, "rows": len(rows), "parts": parts, "stats": stats, "license": source.license,
                              "decision": source.decision, "revision": source.revision, "kind": source.kind}
        S.write_manifest(_done_path(out, source.key), report[source.key])
    for name, rows in sorted((extra_rows or {}).items()):
        cfg = _config_hash(None, {"seed": seed, "dev_percent": dev_percent, "extra": S.digest(S.canonical_json(
            [r["example_id"] + S.request_digest(r["request"]) for r in rows]))})
        done = _already_done(out, name, cfg, resume)
        if done:
            resumed.append(name)
            report[name] = done
            continue
        parts = _write_rows(out, name, rows, dev_percent, shard_rows)
        report[name] = {"config_hash": cfg, "rows": len(rows), "parts": parts, "stats": {"converted": len(rows)},
                        "license": sorted({r["license"] for r in rows}), "decision": "owner", "revision": None, "kind": "extra"}
        S.write_manifest(_done_path(out, name), report[name])
    parts = [p for k in sorted(report) for p in report[k]["parts"]]
    removed = _prune_stale(out, (p["name"] for p in parts), (_done_path(out, k).name for k in report))
    S.assert_unique_example_ids(out / p["name"] for p in parts)
    manifest = {
        "schema_version": S.SCHEMA_VERSION, "kind": "corpus", "seed": seed, "dev_percent": dev_percent,
        "include_review": include_review, "caps_scale": caps_scale, "shard_rows": shard_rows,
        "rows": sum(r["rows"] for r in report.values()),
        "sources": {k: {kk: v[kk] for kk in ("rows", "license", "decision", "revision", "kind", "stats")} for k, v in sorted(report.items())},
        "parts": parts,
        "resumed": sorted(resumed),
        "removed_stale": removed,
    }
    S.write_manifest(out / "manifest.json", manifest)
    return manifest


def make_reader(spec: str) -> Any:
    if spec == "hub":
        return HubReader()
    if spec.startswith("local:"):
        return LocalReader(Path(spec[len("local:"):]))
    raise ValueError("--reader must be 'hub' or 'local:<dir>'")


def inspect(source: SRC.Source, reader: Any, rows: int = 2) -> None:
    from .adapters import ADAPTERS, PREPARERS, AdapterContext

    dataset = reader.open(source)
    print(f"{source.key}: {source.hf_id} config={source.config} split={source.split} license={source.license} decision={source.decision}")
    print("features:", json.dumps(dict(dataset.features), ensure_ascii=False)[:600])
    args = dict(source.args)
    if source.adapter in PREPARERS:
        args.update(PREPARERS[source.adapter](dataset.rows(), args))
    for idx, row in zip(range(rows), dataset.rows()):
        print(f"--- row {idx} fields: {sorted(row)}")
        out = ADAPTERS[source.adapter](row, args, AdapterContext(source.key, idx, DEFAULT_SEED, dataset.features))
        print("converted:", "SKIPPED (check the field names in sources.yaml)" if out is None else json.dumps(
            {"request": out[0], "gold": out[1]}, ensure_ascii=False)[:700])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output directory (parquet parts + manifest.json)")
    ap.add_argument("--reader", default="hub", help="'hub' or 'local:<dir>'")
    ap.add_argument("--sources", type=Path, default=None, help="alternative sources.yaml")
    ap.add_argument("--include-review", action="store_true", help="also use sources marked 'review' (your call, see SCHEMA.md)")
    ap.add_argument("--only", default=None, help="comma-separated source keys")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every cap (e.g. 0.01 for a smoke test)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dev-percent", type=float, default=DEFAULT_DEV_PERCENT)
    ap.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    ap.add_argument("--loop", action="append", default=[], help="Simplicio-Loop JSONL (instruction/context/simplicio_trajectory); repeatable")
    ap.add_argument("--loop-per-record", type=int, default=2)
    ap.add_argument("--inspect", metavar="KEY", help="print the first rows of a source and how they convert, then exit")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    sources = SRC.load_sources(args.sources)
    reader = make_reader(args.reader)
    if args.inspect:
        inspect(sources[args.inspect], reader)
        return 0
    if not args.out:
        ap.error("--out is required")
    extra: Dict[str, List[Dict[str, Any]]] = {}
    if args.loop:
        from .loop_traces import convert_file

        extra["loop"] = [row for path in args.loop for row in convert_file(Path(path), seed=args.seed, per_record=args.loop_per_record,
                                                                          id_prefix=f"loop-{Path(path).parent.name}-{Path(path).stem}")]
    manifest = build_corpus(Path(args.out), sources=sources, reader=reader, seed=args.seed, dev_percent=args.dev_percent,
                            include_review=args.include_review, only=args.only.split(",") if args.only else None,
                            caps_scale=args.scale, shard_rows=args.shard_rows, extra_rows=extra, resume=not args.no_resume)
    print(json.dumps({"rows": manifest["rows"], "sources": {k: v["rows"] for k, v in manifest["sources"].items()},
                      "resumed": manifest["resumed"], "removed_stale": manifest["removed_stale"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
