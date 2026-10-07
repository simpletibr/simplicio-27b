#!/usr/bin/env python3
"""Colab / production serve for Simplicio 27B (issues #4 and #5).

Starts vLLM, puts a completion gate in front of it, opens a Cloudflare
tunnel to the gate, probes a real chat completion, then registers the
tunnel on simpleti.com.br. On exit the upstream is cleared.

A 200 from the gate always has finish_reason in {stop, length, tool_calls}
and a usage object. Completions that would become OpenCode finish=unknown
are rewritten to HTTP 502.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEPLOY = Path(__file__).resolve().parent
ROOT = DEPLOY.parent
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from completion_gate import enforce, json_completion_ok  # noqa: E402

GATEWAY_SET = os.environ.get(
    "SIMPLETI_SET_UPSTREAM_URL",
    "https://simpleti.com.br/api/set_upstream.php",
)
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
GATE_PORT = int(os.environ.get("GATE_PORT", "8001"))
HOST = os.environ.get("HOST", "0.0.0.0")
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


def _http_json(url: str, body: dict[str, Any] | None = None, timeout: int = 30) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SimpleTI-Worker/1.0",
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
            status, payload = _http_json(url, timeout=5)
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


def set_upstream(url: str, key: str) -> Any:
    status, payload = _http_json(f"{GATEWAY_SET}?key={key}", {"upstream_url": url})
    if status >= 400:
        raise RuntimeError(f"set_upstream HTTP {status}: {payload!r}")
    return payload


def clear_upstream(key: str) -> None:
    errors: list[str] = []
    for body in ({"clear": True}, {"upstream_url": None}, {"upstream_url": ""}):
        try:
            status, payload = _http_json(f"{GATEWAY_SET}?key={key}", body)
            if status < 400:
                print("upstream cleared:", payload)
                return
            errors.append(f"{status} {payload!r}")
        except Exception as exc:
            errors.append(str(exc))
    print("clear_upstream failed:", "; ".join(errors), file=sys.stderr)


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
            if key.lower() not in {"host", "content-length"}
        }
        headers["User-Agent"] = headers.get("User-Agent") or "SimpleTI-Worker/1.0"
        target = f"http://127.0.0.1:{VLLM_PORT}{self.path}"
        req = urllib.request.Request(target, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read()
                status = resp.status
                ctype = resp.headers.get("Content-Type") or "application/json"
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
            ctype = exc.headers.get("Content-Type") or "application/json"
        except Exception as exc:
            raw = json.dumps({"error": {"message": str(exc), "type": "GateError"}}).encode()
            status = 502
            ctype = "application/json"
        if self.path.startswith("/v1/chat/completions"):
            status, raw, ctype = enforce(status, raw, ctype)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        self._passthrough("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._passthrough("POST")


def start_gate() -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((HOST, GATE_PORT), _GateHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"completion gate on :{GATE_PORT} → vLLM :{VLLM_PORT}")
    return httpd


def start_cloudflared(port: int) -> tuple[subprocess.Popen[str], str]:
    bin_name = os.environ.get("CLOUDFLARED_BIN", "cloudflared")
    proc = subprocess.Popen(
        [bin_name, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stderr is not None
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stderr.readline()
        if not line and proc.poll() is not None:
            break
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            print("Túnel:", url)
            return proc, url
    raise SystemExit("cloudflared não publicou URL")


def main() -> None:
    load_context()
    key = os.environ.get("SIMPLETI_ADMIN_KEY") or input("Chave admin do gateway: ").strip()
    if not key:
        raise SystemExit("SIMPLETI_ADMIN_KEY ausente")

    vllm_log = VLLM_LOG.open("w", encoding="utf-8")
    vllm = subprocess.Popen(serve_cmd(), stdout=vllm_log, stderr=subprocess.STDOUT)
    gate = None
    tunnel = None
    registered = False

    def shutdown(*_args: Any) -> None:
        nonlocal registered
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

    try:
        wait_vllm(vllm, VLLM_LOG)
        gate = start_gate()
        probe_all(f"http://127.0.0.1:{GATE_PORT}")
        tunnel, public_url = start_cloudflared(GATE_PORT)
        probe_all(public_url)
        print("set_upstream:", set_upstream(public_url, key))
        registered = True
        print("gateway aponta para", public_url)
        while True:
            if vllm.poll() is not None:
                raise SystemExit(f"vLLM saiu com {vllm.returncode}")
            if tunnel.poll() is not None:
                raise SystemExit(f"cloudflared saiu com {tunnel.returncode}")
            time.sleep(5)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
