import argparse
import hashlib
import http.client
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SYSTEM_PROMPT = (
    "Voce e o Simplicio 27B, treinado para executar tarefas de desenvolvimento "
    "seguindo rigorosamente os 50 pontos do Simplicio-Loop: Orientacao, Planejamento, "
    "Edicao Cirurgica por Diff, Validacao e Entrega Verificada sem alucinacao."
)
USER_TEMPLATE = "Contexto: {context}\nArquivo: {edit_file}\nCodigo Atual:\n{code}\nTarefa: {instruction}"
FORMAT_HINT = (
    "\n\nResponda com um ou mais blocos SEARCH/REPLACE neste formato exato, "
    "cada marcador sozinho na linha:\n<<<< SEARCH\n<trecho exato do codigo atual>\n"
    "====\n<trecho novo>\n>>>> REPLACE"
)
SEARCH_RE = re.compile(r"^<{4,7} SEARCH\s*$")
DIVIDER_RE = re.compile(r"^={4,7}\s*$")
REPLACE_RE = re.compile(r"^>{4,7} REPLACE\s*$")
BOOT = """\
import socket, sys
if sys.platform.startswith("linux"):
    import resource
    for limit, value in ((resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_CPU, 70), (resource.RLIMIT_FSIZE, 64 << 20)):
        resource.setrlimit(limit, (value, value))
def _blocked(*args, **kwargs):
    raise OSError("network disabled by eval harness")
socket.socket.connect = socket.socket.connect_ex = _blocked
socket.create_connection = socket.getaddrinfo = _blocked
import pytest
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "tests"]))
"""
TAIL_CHARS = 2000


class PatchError(Exception):
    """Patch application error."""


def load_tasks(tasks_dir):
    tasks = []
    seen = {}
    for d in sorted(Path(tasks_dir).iterdir()):
        if not (d.is_dir() and (d / "task.json").is_file()):
            continue
        task = json.loads((d / "task.json").read_text(encoding="utf-8"))
        if task.get("id") != d.name:
            raise ValueError(f"task id {task.get('id')!r} != directory name {d.name!r}")
        if not (d / "files" / task["edit_file"]).is_file():
            raise ValueError(f"{d.name}: missing files/{task['edit_file']}")
        if not list((d / "tests").glob("test_*.py")):
            raise ValueError(f"{d.name}: missing tests/test_*.py")
        if not (d / "oracle.txt").is_file():
            raise ValueError(f"{d.name}: missing oracle.txt")
        if (d / "files" / "tests").exists():
            raise ValueError(f"{d.name}: files/tests must not exist")
        h = hashlib.sha256()
        for x in (task["instruction"], task["context"], task["edit_file"]):
            h.update(x.encode() + b"\0")
        for root in (d / "files", d / "tests"):
            for p in sorted(p for p in root.rglob("*") if p.is_file()):
                h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
        digest = h.hexdigest()
        if digest in seen:
            raise ValueError(f"duplicate task content: {seen[digest]} and {d.name}")
        seen[digest] = d.name
        tasks.append({**task, "dir": d, "sha256": digest})
    return tasks


def _source(task):
    return (Path(task["dir"]) / "files" / task["edit_file"]).read_text(encoding="utf-8")


def build_messages(task, system, format_hint):
    user = USER_TEMPLATE.format(
        context=task["context"],
        edit_file=task["edit_file"],
        code=_source(task),
        instruction=task["instruction"],
    )
    if format_hint:
        user += FORMAT_HINT
    messages = []
    if system == "simplicio":
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user})
    return messages


def parse_blocks(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = []
    state = "out"
    search = []
    replace = []
    for line in text.splitlines():
        if state == "out":
            if SEARCH_RE.match(line):
                state = "search"
                search, replace = [], []
        elif state == "search":
            if DIVIDER_RE.match(line):
                state = "replace"
            else:
                search.append(line)
        else:
            if REPLACE_RE.match(line):
                blocks.append(("\n".join(search), "\n".join(replace)))
                state = "out"
            else:
                replace.append(line)
    return blocks


def apply_blocks(source, blocks):
    if not blocks:
        raise PatchError("no_blocks")
    for i, (search, replace) in enumerate(blocks):
        if not search:
            raise PatchError(f"empty_search in block {i}")
        if search not in source:
            raise PatchError(f"search_not_found in block {i}")
        source = source.replace(search, replace, 1)
    return source


def sandbox_env(workdir):
    proxy = "http://127.0.0.1:9"
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(workdir),
        "TASK_SRC": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "NO_PROXY": "",
        "no_proxy": "",
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "ALL_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
        "all_proxy": proxy,
    }


def run_tests(workdir, timeout=60):
    kwargs = {}
    if os.name == "nt":
        env = sandbox_env(workdir)
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    else:
        env = sandbox_env(workdir)
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [sys.executable, "-c", BOOT],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs,
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    return {
        "returncode": None if timed_out else proc.returncode,
        "timed_out": timed_out,
        "stdout_tail": (out or "")[-TAIL_CHARS:],
    }


def check_source(task, source, timeout=60):
    with tempfile.TemporaryDirectory(prefix="simplicio-eval-") as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(Path(task["dir"]) / "files", work)
        (work / task["edit_file"]).write_text(source, encoding="utf-8", newline="")
        shutil.copytree(Path(task["dir"]) / "tests", work / "tests")
        result = run_tests(work, timeout)
    result["passed"] = result["returncode"] == 0
    return result


def evaluate(task, raw_output, timeout=60):
    blocks = parse_blocks(raw_output)
    result = {
        "applied": False,
        "apply_error": None,
        "blocks": len(blocks),
        "passed": False,
        "returncode": None,
        "timed_out": False,
        "stdout_tail": "",
    }
    try:
        patched = apply_blocks(_source(task), blocks)
    except PatchError as exc:
        result["apply_error"] = str(exc)
        return result
    result["applied"] = True
    result.update(check_source(task, patched, timeout))
    return result


def chat(base_url, model, messages, api_key, max_tokens, chat_template_kwargs, extra_body, timeout):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    if extra_body is not None:
        body.update(extra_body)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage") or {}
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def wilson_interval(k, n, z=1.959963984540054):
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (0.0 if k == 0 else c - h, 1.0 if k == n else c + h)


def run_oracle(tasks, timeout):
    bad = 0
    for task in tasks:
        if check_source(task, _source(task), timeout)["passed"]:
            print(f"BAD {task['id']}: tests pass without the patch")
            bad += 1
            continue
        oracle = (Path(task["dir"]) / "oracle.txt").read_text(encoding="utf-8")
        res = evaluate(task, oracle, timeout)
        if not res["passed"]:
            print(f"BAD {task['id']}: oracle fails: {res['apply_error'] or res['stdout_tail'][-300:]}")
            bad += 1
        else:
            print(f"OK {task['id']}")
    return 1 if bad else 0


def main(argv=None):
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Run SEARCH/REPLACE patches against hidden pytest tests.")
    ap.add_argument("--tasks", type=Path, default=here / "tasks")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key-env", default="SIMPLETI_API_KEY")
    ap.add_argument("--chat-template-kwargs", type=json.loads, default=None)
    ap.add_argument("--extra-body", type=json.loads, default=None)
    ap.add_argument("--system", choices=["simplicio", "none"], default="simplicio")
    ap.add_argument("--format-hint", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--request-timeout", type=float, default=600)
    ap.add_argument("--test-timeout", type=float, default=60)
    ap.add_argument("--out-dir", type=Path, default=here / "results")
    ap.add_argument("--run-name")
    args = ap.parse_args(argv)

    started = datetime.now(timezone.utc)
    tasks = load_tasks(args.tasks)
    if args.oracle:
        return run_oracle(tasks, args.test_timeout)
    if not args.base_url or not args.model:
        ap.error("--base-url and --model are required unless --oracle")
    run = args.run_name or (
        started.strftime("%Y%m%dT%H%M%SZ") + "-" + re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env)

    n_applied = n_passed = n_errors = 0
    with open(args.out_dir / f"{run}.jsonl", "w", encoding="utf-8") as fh:
        for task in tasks:
            row = {"task": task["id"], "task_sha256": task["sha256"]}
            try:
                messages = build_messages(task, args.system, args.format_hint)
                t0 = time.monotonic()
                reply = chat(
                    args.base_url, args.model, messages, api_key, args.max_tokens,
                    args.chat_template_kwargs, args.extra_body, args.request_timeout,
                )
                latency = round(time.monotonic() - t0, 3)
                row.update({
                    "error": None,
                    "latency_s": latency,
                    "raw_output": reply["content"],
                    "reasoning": reply["reasoning"],
                    "finish_reason": reply["finish_reason"],
                    "prompt_tokens": reply["prompt_tokens"],
                    "completion_tokens": reply["completion_tokens"],
                })
                row.update(evaluate(task, reply["content"], args.test_timeout))
            except (OSError, http.client.HTTPException, KeyError, IndexError, ValueError) as exc:
                row = {
                    "task": task["id"],
                    "task_sha256": task["sha256"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output": None,
                    "applied": False,
                    "passed": False,
                }
                n_errors += 1
            n_applied += bool(row["applied"])
            n_passed += bool(row["passed"])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{task['id']}: {'PASS' if row['passed'] else 'FAIL'}")

    n = len(tasks)
    lo, hi = wilson_interval(n_passed, n)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    summary = {
        "run": run,
        "model": args.model,
        "base_url": args.base_url,
        "system": args.system,
        "format_hint": args.format_hint,
        "chat_template_kwargs": args.chat_template_kwargs,
        "extra_body": args.extra_body,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "test_timeout_s": args.test_timeout,
        "n_tasks": n,
        "applied": n_applied,
        "passed": n_passed,
        "errors": n_errors,
        "pass_rate": n_passed / n,
        "wilson_95": [lo, hi],
        "prompt_sha256": hashlib.sha256(
            (SYSTEM_PROMPT + "\0" + USER_TEMPLATE + "\0" + FORMAT_HINT).encode()
        ).hexdigest(),
        "task_ids": [t["id"] for t in tasks],
        "git_commit": git_commit,
        "python": platform.python_version(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / f"{run}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"passed {n_passed}/{n}  wilson95 [{lo:.3f}, {hi:.3f}]  errors {n_errors}")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
