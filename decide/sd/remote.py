"""Read individual tensors from safetensors files on a Hugging Face-compatible endpoint.

Everything goes through HTTP Range requests: the 8-byte length prefix and the JSON header are
parsed first, then only the byte ranges of the wanted tensors are fetched (adjacent tensors are
coalesced into windows).  No shard is ever downloaded whole, so a 55 GB model can be merged on a
machine with a few GB of disk.  Transient errors are retried with exponential backoff and a
truncated body is resumed from the byte where it stopped.
"""
from __future__ import annotations

import json
import random
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

# safetensors dtype -> bytes per element (static: no torch needed to parse headers or move bytes)
DTYPE_SIZES: Dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}
_TORCH_NAMES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "I64": "int64", "I32": "int32",
    "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}


def torch_dtype(name: str):
    """torch dtype of a safetensors dtype string (imports torch on first use)."""
    import torch

    return getattr(torch, _TORCH_NAMES[name])


RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_HEADER = 256 << 20


class RemoteError(RuntimeError):
    pass


class TensorInfo(NamedTuple):
    dtype: str
    shape: tuple
    start: int       # absolute byte offset in the file
    end: int         # exclusive

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


# --------------------------------------------------------------------------- transports

class LocalTransport:
    """Transport over local files; `url` is a path."""

    def fetch(self, url: str, start: int, end: int) -> bytearray:
        with open(url, "rb") as f:
            f.seek(start)
            return bytearray(f.read(end - start + 1))

    def get(self, url: str) -> bytes:
        with open(url, "rb") as f:
            return f.read()


class HttpTransport:
    """HTTP Range client with retry/backoff and resume of truncated bodies."""

    def __init__(self, token: Optional[str] = None, max_retries: int = 6, backoff: float = 0.5,
                 max_backoff: float = 30.0, timeout: Tuple[float, float] = (15.0, 120.0),
                 sleep: Callable[[float], None] = time.sleep, seed: int = 0) -> None:
        self.token = token
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.max_backoff = float(max_backoff)
        self.timeout = timeout
        self.sleep = sleep
        self._rng = random.Random(seed)
        self._local = threading.local()

    # -- plumbing
    def _session(self):
        import requests

        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {"User-Agent": "simplicio-decide/0.1"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if extra:
            h.update(extra)
        return h

    def _delay(self, attempt: int, retry_after: Optional[str] = None) -> float:
        d = min(self.max_backoff, self.backoff * (2 ** attempt)) * (1.0 + 0.25 * self._rng.random())
        if retry_after:
            try:
                d = max(d, min(self.max_backoff, float(retry_after)))
            except ValueError:
                pass
        return d

    @staticmethod
    def _retryable_exceptions():
        import requests

        return (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError)

    # -- public
    def fetch(self, url: str, start: int, end: int) -> bytearray:
        """Bytes [start, end] inclusive.  A range running past EOF is clipped (Content-Range tells where)."""
        if start < 0 or end < start:
            raise ValueError(f"bad range {start}-{end}")
        buf = bytearray()
        want_end = end
        attempt = 0
        retryable = self._retryable_exceptions()
        while start + len(buf) <= want_end:
            cur = start + len(buf)
            retry_after = None
            try:
                resp = self._session().get(url, headers=self._headers({"Range": f"bytes={cur}-{want_end}"}),
                                           stream=True, timeout=self.timeout, allow_redirects=True)
                code = resp.status_code
                if code == 206:
                    cr = resp.headers.get("Content-Range", "")
                    if cr.startswith("bytes ") and "/" in cr:
                        span, _, total = cr[6:].partition("/")
                        a = int(span.split("-")[0])
                        if a != cur:
                            raise RemoteError(f"server answered range starting at {a}, wanted {cur} ({url})")
                        if total.isdigit():
                            want_end = min(want_end, int(total) - 1)
                    before = len(buf)
                    try:
                        for chunk in resp.iter_content(1 << 20):
                            buf += chunk
                    finally:
                        resp.close()
                    if len(buf) > want_end - start + 1:
                        del buf[want_end - start + 1:]
                    if len(buf) > before:
                        attempt = 0
                    if start + len(buf) <= want_end:
                        raise ConnectionError("short body")   # truncated: resume
                    break
                if code == 200:
                    resp.close()
                    raise RemoteError(f"server ignored the Range header (HTTP 200) for {url}; refusing to download the whole file")
                retry_after = resp.headers.get("Retry-After")
                resp.close()
                if code in RETRY_STATUS:
                    raise _Retry(f"HTTP {code}")
                if code in (401, 403):
                    raise RemoteError(f"HTTP {code} for {url} (token missing/invalid, or the repo is gated or private)")
                raise RemoteError(f"HTTP {code} for {url}")
            except RemoteError:
                raise
            except (_Retry, ConnectionError, *retryable) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise RemoteError(f"{exc} for {url} after {self.max_retries} retries") from exc
                self.sleep(self._delay(attempt - 1, retry_after))
        return buf

    def get(self, url: str) -> bytes:
        """Whole small file (config.json, index.json ...)."""
        attempt = 0
        retryable = self._retryable_exceptions()
        while True:
            retry_after = None
            try:
                resp = self._session().get(url, headers=self._headers(), timeout=self.timeout, allow_redirects=True)
                code = resp.status_code
                if code == 200:
                    return resp.content
                retry_after = resp.headers.get("Retry-After")
                if code in RETRY_STATUS:
                    raise _Retry(f"HTTP {code}")
                if code in (401, 403):
                    raise RemoteError(f"HTTP {code} for {url} (token missing/invalid, or the repo is gated or private)")
                raise RemoteError(f"HTTP {code} for {url}")
            except RemoteError:
                raise
            except (_Retry, *retryable) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise RemoteError(f"{exc} for {url} after {self.max_retries} retries") from exc
                self.sleep(self._delay(attempt - 1, retry_after))


class _Retry(Exception):
    pass


# --------------------------------------------------------------------------- safetensors over Range

class SafetensorsFile:
    """One safetensors file addressed by URL (or path, with LocalTransport)."""

    def __init__(self, url: str, transport, probe: int = 1 << 18) -> None:
        self.url = url
        self.transport = transport
        first = bytes(transport.fetch(url, 0, probe - 1))
        if len(first) < 8:
            raise RemoteError(f"{url}: file too short to be safetensors")
        (n,) = struct.unpack("<Q", first[:8])
        if not 0 < n < MAX_HEADER:
            raise RemoteError(f"{url}: implausible safetensors header length {n}")
        if 8 + n > len(first):
            first += bytes(transport.fetch(url, len(first), 8 + n - 1))
        header = json.loads(first[8:8 + n].decode("utf-8"))
        self.metadata = header.pop("__metadata__", {}) or {}
        base = 8 + n
        self.data_start = base
        self._info: "OrderedDict[str, TensorInfo]" = OrderedDict()
        for name, h in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if h["dtype"] not in DTYPE_SIZES:
                raise RemoteError(f"{url}: unsupported dtype {h['dtype']} for {name}")
            a, b = h["data_offsets"]
            self._info[name] = TensorInfo(h["dtype"], tuple(h["shape"]), base + a, base + b)

    def names(self) -> List[str]:
        return list(self._info)

    def info(self, name: str) -> TensorInfo:
        return self._info[name]

    def __contains__(self, name: str) -> bool:
        return name in self._info

    @staticmethod
    def _to_tensor(buf, info: TensorInfo, shape: tuple):
        import torch

        dtype = torch_dtype(info.dtype)
        n = 1
        for d in shape:
            n *= d
        if n == 0:
            return torch.empty(shape, dtype=dtype)
        if not isinstance(buf, bytearray):
            buf = bytearray(buf)
        return torch.frombuffer(buf, dtype=dtype).reshape(shape)

    def _row_range(self, name: str, row_start: Optional[int], row_end: Optional[int]):
        info = self._info[name]
        if row_start is None and row_end is None:
            return info, info.start, info.end, info.shape
        if not info.shape:
            raise ValueError(f"{name} is a scalar; cannot slice rows")
        rows = info.shape[0]
        r0 = 0 if row_start is None else row_start
        r1 = rows if row_end is None else row_end
        if not 0 <= r0 <= r1 <= rows:
            raise IndexError(f"rows {r0}:{r1} out of range for {name} with {rows} rows")
        row_bytes = info.nbytes // rows if rows else 0
        return info, info.start + r0 * row_bytes, info.start + r1 * row_bytes, (r1 - r0, *info.shape[1:])

    def read_raw(self, name: str, row_start: Optional[int] = None, row_end: Optional[int] = None) -> bytearray:
        """The bytes of a tensor (or of rows row_start:row_end of it), exactly as stored."""
        info, a, b, shape = self._row_range(name, row_start, row_end)
        if b == a:
            return bytearray()
        buf = self.transport.fetch(self.url, a, b - 1)
        if len(buf) != b - a:
            raise RemoteError(f"{self.url}: short read for {name} ({len(buf)} of {b - a} bytes)")
        return buf

    def read_tensor(self, name: str, row_start: Optional[int] = None, row_end: Optional[int] = None):
        info, _a, _b, shape = self._row_range(name, row_start, row_end)
        return self._to_tensor(self.read_raw(name, row_start, row_end), info, shape)

    def plan_windows(self, names: Sequence[str], window_bytes: int, max_gap: int) -> List[Tuple[int, int, List[str]]]:
        """Group tensors into (start, end_exclusive, [names]) windows, in file order."""
        order = sorted(set(names), key=lambda n: self._info[n].start)
        windows: List[Tuple[int, int, List[str]]] = []
        for n in order:
            i = self._info[n]
            if windows:
                s, e, members = windows[-1]
                if i.start - e <= max_gap and (i.end - s) <= window_bytes:
                    windows[-1] = (s, max(e, i.end), members + [n])
                    continue
            windows.append((i.start, i.end, [n]))
        return windows

    def fetch_window(self, window: Tuple[int, int, List[str]]) -> Dict[str, Any]:
        s, e, members = window
        buf = self.transport.fetch(self.url, s, e - 1) if e > s else bytearray()
        if len(buf) != e - s:
            raise RemoteError(f"{self.url}: short window read ({len(buf)} of {e - s})")
        view = memoryview(buf)
        out = {}
        for n in members:
            i = self._info[n]
            chunk = bytearray(view[i.start - s:i.end - s])   # own the memory so the window can be freed
            out[n] = self._to_tensor(chunk, i, i.shape)
        return out


# --------------------------------------------------------------------------- a repo

class HubRepo:
    """A model repo read through Range requests.  `endpoint` is overridable for tests."""

    def __init__(self, repo_id: str, revision: Optional[str] = None, token: Optional[str] = None,
                 endpoint: str = "https://huggingface.co", transport=None, probe: int = 1 << 18, **transport_kw) -> None:
        self.probe = probe
        self.repo_id = repo_id
        self.revision = revision or "main"
        self.endpoint = endpoint.rstrip("/")
        self.transport = transport or HttpTransport(token=token, **transport_kw)
        self._shards: Dict[str, SafetensorsFile] = {}
        self._weight_map: Optional[Dict[str, str]] = None
        self._lock = threading.RLock()

    def url(self, filename: str) -> str:
        return f"{self.endpoint}/{self.repo_id}/resolve/{self.revision}/{filename}"

    def read_bytes(self, filename: str) -> bytes:
        return self.transport.get(self.url(filename))

    def read_json(self, filename: str):
        return json.loads(self.read_bytes(filename).decode("utf-8"))

    def exists(self, filename: str) -> bool:
        try:
            self.transport.fetch(self.url(filename), 0, 0)
            return True
        except RemoteError as exc:
            if "404" in str(exc):
                return False
            raise

    def shard(self, filename: str) -> SafetensorsFile:
        with self._lock:
            sf = self._shards.get(filename)
            if sf is None:
                sf = SafetensorsFile(self.url(filename), self.transport, self.probe)
                self._shards[filename] = sf
            return sf

    def weight_map(self) -> Dict[str, str]:
        with self._lock:
            if self._weight_map is None:
                try:
                    self._weight_map = dict(self.read_json("model.safetensors.index.json")["weight_map"])
                except RemoteError as exc:
                    if "404" not in str(exc):
                        raise
                    sf = self.shard("model.safetensors")
                    self._weight_map = {n: "model.safetensors" for n in sf.names()}
            return self._weight_map

    def names(self) -> List[str]:
        return list(self.weight_map())

    def shard_of(self, name: str) -> str:
        return self.weight_map()[name]

    def info(self, name: str) -> TensorInfo:
        return self.shard(self.shard_of(name)).info(name)

    def read_tensor(self, name: str, row_start: Optional[int] = None, row_end: Optional[int] = None):
        return self.shard(self.shard_of(name)).read_tensor(name, row_start, row_end)

    def file_order(self, names: Iterable[str]) -> List[str]:
        """Sort names by (shard, offset): the cheapest order to stream them."""
        def key(n):
            s = self.shard_of(n)
            return (s, self.shard(s).info(n).start)
        return sorted(names, key=key)

    def stream(self, names: Sequence[str], window_bytes: int = 256 << 20, workers: int = 2,
               max_gap: int = 1 << 20) -> Iterator[Tuple[str, Any]]:
        """Yield (name, tensor) in the order of `names`, prefetching coalesced windows in threads.

        Pass names in file order (see `file_order`) to keep memory bounded to ~ (workers+1) windows.
        """
        names = list(names)
        by_shard: Dict[str, List[str]] = {}
        for n in names:
            by_shard.setdefault(self.shard_of(n), []).append(n)
        windows: List[Tuple[str, Tuple[int, int, List[str]]]] = []
        where: Dict[str, int] = {}
        for shard_name, members in by_shard.items():
            for w in self.shard(shard_name).plan_windows(members, window_bytes, max_gap):
                for m in w[2]:
                    where[m] = len(windows)
                windows.append((shard_name, w))
        # windows in the order of first use
        first_use: Dict[int, int] = {}
        for pos, n in enumerate(names):
            first_use.setdefault(where[n], pos)
        schedule = sorted(first_use, key=first_use.get)
        futures: Dict[int, Future] = {}
        cache: Dict[int, Dict[str, Any]] = {}
        remaining = {i: len(windows[i][1][2]) for i in schedule}
        nxt = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            def submit_more(upto: int) -> None:
                nonlocal nxt
                while nxt < len(schedule) and nxt <= upto:
                    wi = schedule[nxt]
                    shard_name, w = windows[wi]
                    futures[wi] = pool.submit(self.shard(shard_name).fetch_window, w)
                    nxt += 1

            for n in names:
                wi = where[n]
                pos_in_schedule = schedule.index(wi) if wi not in futures and wi not in cache else None
                submit_more((pos_in_schedule if pos_in_schedule is not None else nxt) + max(1, workers))
                if wi not in cache:
                    cache[wi] = futures.pop(wi).result()
                t = cache[wi].pop(n)
                remaining[wi] -= 1
                if remaining[wi] == 0:
                    cache.pop(wi, None)
                yield n, t
