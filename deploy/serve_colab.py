#!/usr/bin/env python3
"""Colab / production serve for Simplicio 27B (issues #4 and #5).

Starts vLLM, puts a completion gate in front of it, opens a Cloudflare
tunnel to the gate, probes a real chat completion, then registers the
tunnel on simpleti.com.br and re-registers it every 30 s (heartbeat): the gateway's
lease expires 90 s after the last beat, so a dead Colab ends in HTTP 503. On exit the upstream is cleared.

A non-stream 200 from the gate always has finish_reason in {stop, length,
tool_calls} and a usage object; otherwise it becomes HTTP 502. Streams are
forwarded line by line and end with an SSE error event when finish_reason
is missing.
"""

from __future__ import annotations

import getpass
import hmac
import http.client
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEPLOY = Path(__file__).resolve().parent
ROOT = DEPLOY.parent
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from completion_gate import (  # noqa: E402
    SSE_DONE,
    StreamCheck,
    enforce,
    json_completion_ok,
    sse_error_event,
)

GATEWAY_SET = os.environ.get(
    "SIMPLETI_SET_UPSTREAM_URL",
    "https://simpleti.com.br/api/set_upstream.php",
)
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
GATE_PORT = int(os.environ.get("GATE_PORT", "8001"))
HEARTBEAT_S = 30
LOCAL = "127.0.0.1"
UPSTREAM_HEADER = "X-Simpleti-Upstream-Token"
GATE_ROUTES = {("GET", "/v1/models"), ("POST", "/v1/chat/completions")}
# Per-run secrets. Never print them and never write them to disk.
VLLM_TOKEN = secrets.token_urlsafe(32)  # only the gate sends it to vLLM
GATE_TOKEN = secrets.token_urlsafe(32)  # only the gateway sends it to the gate
MODEL_ID = os.environ.get("MODEL_ID", "wesleysimplicio/Simplicio-27B")
SERVED_IDS = ("simplicio-27b", "simpleti/simplicio-27b")
VLLM_LOG = Path("vllm.log")


def load_context(path: Path | None = None) -> dict[str, int]:
    env_path = path or (DEPLOY / "context.env")
    values: dict[str, int] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        values[key.strip()] = int(raw.strip())
    for required in ("MAX_MODEL_LEN", "OUTPUT_BUDGET", "MEASURED_AGENT_PROMPT"):
        if required not in values:
            raise RuntimeError(f"missing {required} in {env_path}")
    if values["MAX_MODEL_LEN"] < values["MEASURED_AGENT_PROMPT"] + values["OUTPUT_BUDGET"]:
        raise RuntimeError(
            "MAX_MODEL_LEN must be >= MEASURED_AGENT_PROMPT + OUTPUT_BUDGET"
        )
    return values


def serve_cmd() -> list[str]:
    """The only vLLM launch: deploy/serve_vllm.sh holds every flag."""
    return ["bash", str(DEPLOY / "serve_vllm.sh"), MODEL_ID, str(VLLM_PORT)]


# What vLLM may inherit: paths, locale, proxy/CA settings for the weights download, GPU libraries and its own
# variables (the HF_ ones include the Hugging Face token for private weights; the gateway admin key is not here).
VLLM_ENV_NAMES = frozenset(
    {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "TMPDIR", "TZ", "LANG", "LANGUAGE",
        "LD_LIBRARY_PATH", "LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONHASHSEED",
        "VIRTUAL_ENV", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "GPU_MEM", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE", "TRANSFORMERS_CACHE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "no_proxy", "all_proxy",
    }
)
VLLM_ENV_PREFIXES = ("LC_", "CUDA_", "NVIDIA_", "NCCL_", "VLLM_", "HF_", "TORCH_", "TRITON_", "FLASHINFER_")


def serve_env() -> dict[str, str]:
    """vLLM listens only on loopback, requires VLLM_TOKEN on /v1/* and sees no other credential."""
    inherited = {
        name: value
        for name, value in os.environ.items()
        if name in VLLM_ENV_NAMES or name.startswith(VLLM_ENV_PREFIXES)
    }
    return {**inherited, "HOST": LOCAL, "VLLM_API_KEY": VLLM_TOKEN}


def tunnel_env() -> dict[str, str]:
    """cloudflared needs no gateway credential: it inherits the environment without any SIMPLETI_* variable."""
    return {name: value for name, value in os.environ.items() if not name.startswith("SIMPLETI_")}


def _http_json(
    url: str,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SimpleTI-Worker/1.0",
            **(headers or {}),
        },
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                parsed = raw.decode("utf-8", errors="replace")
            return resp.status, parsed
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = raw.decode("utf-8", errors="replace")
        return exc.code, parsed


def tail_log(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(sem {path}: {exc})"
    return "\n".join(text.splitlines()[-lines:])


def wait_vllm(proc: subprocess.Popen[Any], log_path: Path, timeout_s: int = 1200) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{VLLM_PORT}/v1/models"
    last = ""
    while time.time() < deadline:
        code = proc.poll()
        if code is not None:
            raise SystemExit(
                f"vLLM saiu com código {code} antes de ficar pronto. "
                f"Últimas 40 linhas de {log_path}:\n{tail_log(log_path)}"
            )
        try:
            status, payload = _http_json(
                url, timeout=5, headers={"Authorization": f"Bearer {VLLM_TOKEN}"}
            )
            if status == 200:
                print("vLLM pronto")
                return
            last = repr(payload)
        except Exception as exc:
            last = str(exc)
        time.sleep(5)
    raise SystemExit(f"vLLM não subiu em {timeout_s}s: {last}\n{tail_log(log_path)}")


def probe_completion(base: str, model: str) -> dict[str, Any]:
    """POST a tiny chat completion. Raises if finish_reason/usage are missing."""
    status, payload = _http_json(
        f"{base.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 64,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with exactly pong."}],
        },
        timeout=180,
        headers={UPSTREAM_HEADER: GATE_TOKEN},
    )
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(f"probe {model} HTTP {status}: {payload!r}")
    err = json_completion_ok(payload)
    if err:
        raise RuntimeError(f"probe {model} rejected: {err}")
    message = payload["choices"][0].get("message") or {}
    if not (message.get("content") or "").strip():
        raise RuntimeError(f"probe {model} rejected: empty content")
    return payload


def probe_all(base: str) -> None:
    for model in SERVED_IDS:
        probe_completion(base, model)
        print(f"probe ok: {model} via {base}")


def admin_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def read_admin_key() -> str:
    key = os.environ.get("SIMPLETI_ADMIN_KEY") or getpass.getpass("Chave admin do gateway: ")
    key = key.strip()
    if not key:
        raise SystemExit("SIMPLETI_ADMIN_KEY ausente")
    return key


def set_upstream(url: str, key: str) -> Any:
    status, payload = _http_json(
        GATEWAY_SET,
        {"upstream_url": url, "upstream_token": GATE_TOKEN},
        headers=admin_headers(key),
    )
    if status >= 400:
        raise RuntimeError(f"set_upstream HTTP {status}: {payload!r}")
    return payload


def clear_upstream(key: str) -> None:
    errors: list[str] = []
    for body in ({"clear": True}, {"upstream_url": None}, {"upstream_url": ""}):
        try:
            status, payload = _http_json(GATEWAY_SET, body, headers=admin_headers(key))
            if status < 400:
                print("upstream cleared:", payload)
                return
            errors.append(f"{status} {payload!r}")
        except Exception as exc:
            errors.append(str(exc))
    print("clear_upstream failed:", "; ".join(errors), file=sys.stderr)


def _upstream_lines(resp: http.client.HTTPResponse):
    """Yield upstream SSE lines; stop quietly if the upstream drops mid-stream."""
    while True:
        try:
            line = resp.readline()
        except (OSError, http.client.HTTPException) as exc:
            sys.stderr.write(f"gate: upstream caiu no meio do stream: {exc!r}\n")
            return
        if not line:
            return
        yield line


class _GateHandler(BaseHTTPRequestHandler):
    server_version = "SimplicioCompletionGate/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("gate: " + (fmt % args) + "\n")

    def _passthrough(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {
            key: val
            for key, val in self.headers.items()
            if key.lower() not in {"host", "content-length", "authorization", UPSTREAM_HEADER.lower()}
        }
        headers["User-Agent"] = headers.get("User-Agent") or "SimpleTI-Worker/1.0"
        headers["Authorization"] = f"Bearer {VLLM_TOKEN}"
        target = f"http://127.0.0.1:{VLLM_PORT}{self.path}"
        req = urllib.request.Request(target, data=body, headers=headers, method=method)
        chat = self.path.startswith("/v1/chat/completions")
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
            ctype = exc.headers.get("Content-Type") or "application/json"
        except Exception as exc:
            raw = json.dumps({"error": {"message": str(exc), "type": "GateError"}}).encode()
            status = 502
            ctype = "application/json"
        else:
            with resp:
                ctype = resp.headers.get("Content-Type") or "application/json"
                if "text/event-stream" in ctype.lower():
                    self._stream(resp, ctype, check=chat)
                    return
                raw = resp.read()
                status = resp.status
        if chat:
            status, raw, ctype = enforce(status, raw, ctype)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _stream(self, resp: http.client.HTTPResponse, ctype: str, check: bool) -> None:
        """Forward SSE line by line; end with an error event if the stream was bad."""
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.close_connection = True
        tracker = StreamCheck()
        try:
            for line in _upstream_lines(resp):
                tracker.feed(line)
                if tracker.done:
                    break
                self.wfile.write(line)
                self.wfile.flush()
            problem = tracker.problem() if check else None
            if problem:
                self.log_message("stream inválido: %s", problem)
                self.wfile.write(sse_error_event(problem))
            elif check and tracker.usage is None:
                self.log_message("stream sem usage (falta --enable-force-include-usage?)")
            self.wfile.write(SSE_DONE)
            self.wfile.flush()
        except ConnectionError:
            self.log_message("cliente desconectou; fechando o upstream")
        finally:
            resp.close()

    def _deny(self, status: int, message: str) -> None:
        length = self.headers.get("Content-Length") or "0"
        if length.isascii() and length.isdigit():
            self.rfile.read(min(int(length), 1 << 20))
        raw = json.dumps({"error": {"message": message, "type": "GateError"}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def _allowed(self) -> bool:
        sent = self.headers.get(UPSTREAM_HEADER) or ""
        if not hmac.compare_digest(sent.encode(), GATE_TOKEN.encode()):
            self._deny(401, "missing or invalid upstream token")
            return False
        if (self.command, self.path.split("?", 1)[0]) not in GATE_ROUTES:
            self._deny(404, "not found")
            return False
        if self.headers.get("Transfer-Encoding") is not None:
            # The gate forwards only Content-Length bodies; a chunked request would reach vLLM with no body.
            self._deny(411, "length required")
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if self._allowed():
            self._passthrough("GET")

    def do_POST(self) -> None:  # noqa: N802
        if self._allowed():
            self._passthrough("POST")


def start_gate() -> tuple[ThreadingHTTPServer, threading.Thread]:
    httpd = ThreadingHTTPServer((LOCAL, GATE_PORT), _GateHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"completion gate on :{GATE_PORT} → vLLM :{VLLM_PORT}")
    return httpd, thread


def start_cloudflared(
    port: int, log_path: Path = Path("cloudflared.log"), timeout_s: float = 60
) -> tuple[subprocess.Popen[bytes], str]:
    """Open a Quick Tunnel. cloudflared logs go to a file, so no pipe can fill up and stall it."""
    bin_name = os.environ.get("CLOUDFLARED_BIN", "cloudflared")
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [bin_name, "tunnel", "--url", f"http://127.0.0.1:{port}"],
            env=tunnel_env(),
            stdout=subprocess.DEVNULL,
            stderr=log,
        )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        exited = proc.poll() is not None  # polled before the read, so the last lines of a dead child are seen
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
        if match:
            print("Túnel:", match.group(0))
            return proc, match.group(0)
        if exited:
            break
        time.sleep(0.5)
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
    raise SystemExit(f"cloudflared não publicou URL:\n{tail}")


def probe_public(probe: Callable[[str], Any], url: str, attempts: int = 6, wait_s: float = 10) -> None:
    """A new trycloudflare hostname takes a few seconds to resolve: retry the probe."""
    for attempt in range(1, attempts + 1):
        try:
            probe(url)
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"probe do túnel falhou ({attempt}/{attempts}): {exc}")
            time.sleep(wait_s)


def dead_child(children: dict[str, Callable[[], bool]]) -> str | None:
    for name, alive in children.items():
        if not alive():
            return name
    return None


def vllm_healthy() -> bool:
    try:
        status, _ = _http_json(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=10)
    except Exception:
        return False
    return status == 200


def heartbeat_loop(
    register: Callable[[], Any],
    children: dict[str, Callable[[], bool]],
    stop: threading.Event,
    interval: float = HEARTBEAT_S,
) -> None:
    """register() every `interval` s until `stop` is set or a child dies; skip the beat if vLLM /health is not 200."""
    while not stop.wait(interval):
        dead = dead_child(children)
        if dead is not None:
            print(f"heartbeat parado: {dead} morreu", file=sys.stderr)
            return
        if not vllm_healthy():
            print("heartbeat: /health do vLLM falhou; lease não renovado", file=sys.stderr)
            continue
        try:
            register()
        except Exception as exc:
            print(f"heartbeat: set_upstream falhou: {exc}", file=sys.stderr)


def main() -> None:
    load_context()
    key = read_admin_key()

    vllm_log = VLLM_LOG.open("w", encoding="utf-8")
    vllm = subprocess.Popen(
        serve_cmd(), env=serve_env(), stdout=vllm_log, stderr=subprocess.STDOUT
    )
    gate = None
    tunnel = None
    registered = False
    stop_beat = threading.Event()
    beat: threading.Thread | None = None

    def shutdown(*_args: Any) -> None:
        nonlocal registered
        stop_beat.set()
        if beat is not None and beat.is_alive():
            beat.join(timeout=35)
        if registered:
            clear_upstream(key)
            registered = False
        if tunnel and tunnel.poll() is None:
            tunnel.terminate()
        if gate is not None:
            gate.shutdown()
        if vllm.poll() is None:
            vllm.terminate()
            try:
                vllm.wait(timeout=15)
            except subprocess.TimeoutExpired:
                vllm.kill()

    signal.signal(signal.SIGTERM, lambda *_: (shutdown(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))
    signal.signal(signal.SIGHUP, lambda *_: (shutdown(), sys.exit(0)))

    try:
        wait_vllm(vllm, VLLM_LOG)
        gate, gate_thread = start_gate()
        probe_all(f"http://127.0.0.1:{GATE_PORT}")
        tunnel, public_url = start_cloudflared(GATE_PORT)
        probe_public(probe_all, public_url)
        print("set_upstream:", set_upstream(public_url, key))
        registered = True
        print("gateway aponta para", public_url)
        children = {
            "vllm": lambda: vllm.poll() is None,
            "gate": gate_thread.is_alive,
            "cloudflared": lambda: tunnel.poll() is None,
        }
        beat = threading.Thread(
            target=heartbeat_loop,
            args=(lambda: set_upstream(public_url, key), children, stop_beat),
            name="heartbeat",
            daemon=True,
        )
        beat.start()
        while True:
            dead = dead_child(children)
            if dead is not None:
                raise SystemExit(f"{dead} morreu; limpando o upstream")
            time.sleep(5)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
