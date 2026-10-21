"""Shared helpers for the decide/ kit tests.

Not a test module (unittest discovery only loads test*.py).  Everything here works with the standard library
and numpy; torch, transformers and pyarrow are only touched by the helpers that say so.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import struct
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def have(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAVE_NUMPY = have("numpy")
HAVE_TORCH = have("torch")
HAVE_TRANSFORMERS = HAVE_TORCH and have("transformers")
HAVE_PYARROW = have("pyarrow")
HAVE_KIT = have("decision_index")

needs_numpy = unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
needs_torch = unittest.skipUnless(HAVE_TORCH, "torch is not installed")
needs_transformers = unittest.skipUnless(HAVE_TRANSFORMERS, "torch + transformers are not installed")
needs_pyarrow = unittest.skipUnless(HAVE_PYARROW, "pyarrow is not installed")
needs_kit = unittest.skipUnless(HAVE_KIT, "the decision-index scoring kit is not installed")

SECRET_ENV = ("HF_TOKEN", "SD_PLATFORM", "SD_WORKDIR", "SD_CONFIG", "KAGGLE_KERNEL_RUN_TYPE", "COLAB_RELEASE_TAG")


class HermeticTestCase(unittest.TestCase):
    """Starts every test without tokens or platform hints in the process environment."""

    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in SECRET_ENV:
            os.environ.pop(key, None)


# --------------------------------------------------------------------------- safetensors without the library

def build_safetensors(tensors: Dict[str, Tuple[str, tuple, bytes]], metadata: Optional[dict] = None) -> bytes:
    """tensors: name -> (safetensors dtype, shape, raw little-endian bytes).  Layout per the safetensors spec."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    blobs: List[bytes] = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + b"".join(blobs)


def numpy_tensors(seed: int = 0) -> Dict[str, Tuple[str, tuple, bytes]]:
    """Small named tensors in the dtypes a decision backbone uses (bf16 as random 16-bit payloads)."""
    import numpy as np

    rng = np.random.default_rng(seed)

    def bf16(*shape):
        return ("BF16", shape, rng.integers(0, 2**15, size=shape, dtype=np.uint16).tobytes())

    return {
        "language_model.layers.0.mlp.gate_proj.weight": bf16(24, 16),
        "language_model.layers.0.input_layernorm.weight": bf16(16),
        "language_model.layers.0.linear_attn.A_log": ("F32", (8,), rng.standard_normal(8).astype("<f4").tobytes()),
        "language_model.layers.1.mlp.up_proj.weight": ("F16", (32, 16), rng.standard_normal((32, 16)).astype("<f2").tobytes()),
        "language_model.embed_tokens.weight": bf16(50, 16),
        "visual.blocks.0.attn.proj.bias": bf16(4),
        "mask": ("I64", (3,), np.array([1, 0, 1], dtype="<i8").tobytes()),
        "zz.big": ("BF16", (300_000,), bytes(600_000)),       # ~600 KB: the file is much larger than the header probe
    }


# --------------------------------------------------------------------------- a fake Hugging Face endpoint

class FakeHub:
    """Real HTTP server with Range support and fault injection.  URL layout mimics the Hub:
    /{org}/{name}/resolve/{revision}/{path}
    """

    def __init__(self) -> None:
        self.files: Dict[str, bytes] = {}          # "org/name/path" -> bytes
        self.log: List[Tuple[str, str, int]] = []  # (key, range header, bytes sent)
        self.fail_first: Dict[str, int] = {}       # key -> remaining 503s
        self.truncate_once: Dict[str, int] = {}    # key -> send only N bytes of the body once, then drop the connection
        self.ignore_range: set = set()             # keys served with 200 and the full body
        self.redirect: set = set()                 # keys served through a 302 to /cdn/<key>
        self.auth_seen: List[str] = []
        self.require_token: Optional[str] = None
        self._lock = threading.Lock()
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a, **k):  # silence
                pass

            def _lookup(self):
                path = self.path.split("?")[0]
                m = re.match(r"^/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$", path)
                if m:
                    return path, f"{m.group(1)}/{m.group(3)}", False
                m = re.match(r"^/cdn/(.+)$", path)
                if m:
                    return path, m.group(1), True
                return path, None, False

            def do_GET(self):
                path, key, via_cdn = self._lookup()
                hub.auth_seen.append(self.headers.get("Authorization", ""))
                if hub.require_token and self.headers.get("Authorization") != f"Bearer {hub.require_token}" and not via_cdn:
                    return self._send(401, b"unauthorized")
                if key is None or key not in hub.files:
                    return self._send(404, b"not found")
                with hub._lock:
                    n = hub.fail_first.get(key, 0)
                    if n > 0:
                        hub.fail_first[key] = n - 1
                        return self._send(503, b"busy")
                if key in hub.redirect and not via_cdn:
                    return self._send(302, b"", extra={"Location": f"/cdn/{key}"})
                data = hub.files[key]
                rng = self.headers.get("Range")
                if rng and key not in hub.ignore_range:
                    m = re.match(r"bytes=(\d+)-(\d*)$", rng)
                    if not m:
                        return self._send(416, b"bad range")
                    a = int(m.group(1))
                    b = int(m.group(2)) if m.group(2) else len(data) - 1
                    b = min(b, len(data) - 1)
                    if a >= len(data) or a > b:
                        return self._send(416, b"unsatisfiable", extra={"Content-Range": f"bytes */{len(data)}"})
                    body = data[a:b + 1]
                    with hub._lock:
                        cut = hub.truncate_once.pop(key, None)
                    hub.log.append((key, rng, len(body) if cut is None else min(cut, len(body))))
                    return self._send(206, body, extra={"Content-Range": f"bytes {a}-{b}/{len(data)}"}, cut=cut)
                hub.log.append((key, rng or "", len(data)))
                return self._send(200, data)

            def _send(self, code, body, extra=None, cut=None):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if cut is not None:
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    try:
                        self.connection.shutdown(2)
                    except OSError:
                        pass
                    return
                self.wfile.write(body)

        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request, client_address):  # clients hang up mid-body in fault tests
                pass

        self.server = QuietServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def add(self, repo: str, path: str, data: bytes) -> None:
        self.files[f"{repo}/{path}"] = bytes(data)

    def __enter__(self) -> "FakeHub":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def bytes_served(self, key_prefix: str = "") -> int:
        return sum(n for k, _r, n in self.log if k.startswith(key_prefix))
