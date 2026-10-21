"""Dataset readers: local files (tests, offline, your own data) and the Hugging Face Hub (parquet files).

A reader's ``open(source)`` returns an object with ``features`` and a re-iterable ``rows()``.
The Hub reader needs ``huggingface_hub`` and ``pyarrow`` and the network; it is not exercised by the unit tests
(it was run by hand, see the PR).  It never executes code from a dataset repository: it only downloads parquet.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import posixpath
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from . import schema as S
from .sources import Source

CONVERT_REVISION = "refs/convert/parquet"


class ListDataset:
    def __init__(self, rows: List[Mapping[str, Any]], features: Optional[Mapping[str, Any]] = None) -> None:
        self._rows, self.features = rows, features or {}

    def rows(self) -> Iterator[Mapping[str, Any]]:
        return iter(self._rows)


class ParquetDataset:
    def __init__(self, files: List[Path], features: Mapping[str, Any]) -> None:
        self.files, self.features = files, features

    def rows(self) -> Iterator[Mapping[str, Any]]:
        _, pq = S._arrow()
        for file in self.files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=S.BATCH_ROWS):
                yield from batch.to_pylist()


def parquet_features(path: Path) -> Dict[str, Any]:
    """HF ``features`` stored in the parquet schema metadata (class names live here)."""
    _, pq = S._arrow()
    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return {}
    try:
        return dict(json.loads(raw).get("info", {}).get("features", {}))
    except ValueError:
        return {}


class LocalReader:
    """``root/<key>.jsonl[.gz]``, ``root/<key>.parquet`` or ``root/<key>/*.parquet`` (+ optional ``<key>.features.json``)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def open(self, source: Source) -> ParquetDataset | ListDataset:
        base = self.root / source.key
        feats_file = self.root / f"{source.key}.features.json"
        features = json.loads(feats_file.read_text(encoding="utf-8")) if feats_file.exists() else {}
        for suffix in (".jsonl", ".jsonl.gz"):
            path = base.with_name(base.name + suffix)
            if path.exists():
                opener = gzip.open if suffix.endswith(".gz") else open
                with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[call-overload]
                    return ListDataset([json.loads(line) for line in fh if line.strip()], features)
        files = S.corpus_files(base) if base.is_dir() else [base.with_name(base.name + ".parquet")]
        if not files or not files[0].exists():
            raise FileNotFoundError(f"no local data for source {source.key!r} under {self.root}")
        return ParquetDataset(files, features or parquet_features(files[0]))


def select_parquet_files(files: List[str], config: Optional[str], split: str) -> List[str]:
    """Pick the parquet files of one split (and config) from a repo file listing."""
    cfg = None if config in (None, "default") else config
    chosen = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        base = posixpath.basename(f)
        in_split = base.startswith(f"{split}-") or base == f"{split}.parquet" or split in parts[:-1] or f"partial-{split}" in parts[:-1]
        if not in_split or (cfg is not None and cfg not in parts[:-1]):
            continue
        chosen.append(f)
    tops = {c.split("/")[0] for c in chosen}
    if cfg is None and len(tops) > 1 and not tops <= {"data", "default"}:
        raise ValueError(f"several configs match ({sorted(tops)}); set 'config' in sources.yaml")
    return sorted(chosen)


class UrlReader:
    """Reads a CSV or JSONL file (and an optional JSON list of class names) from a URL, checking pinned sha256 sums.

    For repositories that only ship a loading script (e.g. PolyAI/banking77): the data come from the files the
    script would have downloaded, and the script itself is never run.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache = Path(cache_dir or os.path.expanduser("~/.cache/decide-data"))

    def fetch(self, url: str, sha256: Optional[str] = None) -> bytes:
        if urllib.parse.urlparse(url).scheme not in ("https", "file"):
            raise ValueError(f"only https:// (or file://) URLs are fetched, got {url!r}")
        self.cache.mkdir(parents=True, exist_ok=True)
        cached = self.cache / hashlib.sha256(url.encode("utf-8")).hexdigest()
        data = cached.read_bytes() if cached.exists() else None
        if data is None or (sha256 and hashlib.sha256(data).hexdigest() != sha256):
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - the URL comes from sources.yaml
                data = resp.read()
            if sha256 and hashlib.sha256(data).hexdigest() != sha256:
                raise ValueError(f"sha256 mismatch for {url}: the upstream file changed (expected {sha256})")
            tmp = cached.with_name(cached.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cached)
        return data

    def open(self, source: Source) -> ListDataset:
        files, pins = source.files, source.files.get("sha256") or {}
        raw = self.fetch(files["data"], pins.get("data")).decode("utf-8")
        if files["data"].split("?")[0].endswith(".csv"):
            rows: List[Mapping[str, Any]] = list(csv.DictReader(io.StringIO(raw)))
        else:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        features: Dict[str, Any] = {}
        if files.get("labels"):
            names = json.loads(self.fetch(files["labels"], pins.get("labels")).decode("utf-8"))
            features[str(source.args.get("label"))] = {"names": [str(n) for n in names]}
        return ListDataset(rows, features)


class HubReader:
    """Downloads the parquet files of a split from the Hub (main, or ``refs/convert/parquet`` for script repos)."""

    def __init__(self, token: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        self.token, self.cache_dir = token, cache_dir

    def open(self, source: Source) -> ParquetDataset | ListDataset:
        if source.files:
            return UrlReader(self.cache_dir).open(source)
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=self.token)
        revision = source.revision
        files = select_parquet_files(api.list_repo_files(source.hf_id, repo_type="dataset", revision=revision), source.config, source.split)
        if not files:
            revision = CONVERT_REVISION
            files = select_parquet_files(api.list_repo_files(source.hf_id, repo_type="dataset", revision=revision), source.config, source.split)
        if not files:
            raise FileNotFoundError(f"{source.hf_id}: no parquet files for config={source.config!r} split={source.split!r}")
        local = [Path(hf_hub_download(source.hf_id, f, repo_type="dataset", revision=revision, token=self.token, cache_dir=self.cache_dir)) for f in files]
        return ParquetDataset(local, parquet_features(local[0]))
