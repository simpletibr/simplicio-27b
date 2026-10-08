"""Heartbeat lease, public probe retry and cloudflared log handling (issue #19)."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import serve_colab  # noqa: E402

URL = "https://abc-def.trycloudflare.com"
NOISY = (
    "sys.stderr.write('INF |  https://abc-def.trycloudflare.com  |\\n'); sys.stderr.flush()\n"
    "sys.stderr.write('x' * 262144 + '\\n'); sys.stderr.flush()"
)
ALIVE = lambda: True  # noqa: E731


def fake_cloudflared(tmp: Path, body: str) -> str:
    script = tmp / "cloudflared"
    script.write_text(f"#!{sys.executable}\nimport sys, time\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


class FakeStop:
    """Stands in for threading.Event: wait() is False for the first `ticks` calls."""

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self.waits: list[float] = []

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        return len(self.waits) > self.ticks


class HeartbeatTests(unittest.TestCase):
    def run_loop(self, stop, children, health, effect):
        register = Mock(side_effect=effect)
        with patch.object(serve_colab, "vllm_healthy", side_effect=health * 10), contextlib.redirect_stderr(io.StringIO()):
            serve_colab.heartbeat_loop(register, children, stop)
        return register

    def test_posts_every_30s_while_children_alive(self) -> None:
        stop = FakeStop(3)
        register = self.run_loop(stop, {"vllm": ALIVE, "gate": ALIVE, "cloudflared": ALIVE}, [True], None)
        self.assertEqual(register.call_count, 3)
        self.assertEqual(stop.waits, [30, 30, 30, 30])

    def test_stops_for_good_when_a_child_dies(self) -> None:
        stop = FakeStop(10)
        states = iter([True, False])
        register = self.run_loop(stop, {"vllm": ALIVE, "cloudflared": lambda: next(states)}, [True], None)
        self.assertEqual(register.call_count, 1)
        self.assertEqual(len(stop.waits), 2)

    def test_unhealthy_vllm_skips_the_beat(self) -> None:
        register = self.run_loop(FakeStop(2), {"vllm": ALIVE}, [False, True], None)
        self.assertEqual(register.call_count, 1)

    def test_failed_post_does_not_stop_the_loop(self) -> None:
        register = self.run_loop(FakeStop(2), {"vllm": ALIVE}, [True], [RuntimeError("HTTP 500"), None])
        self.assertEqual(register.call_count, 2)

    def test_dead_child_names_the_first_dead(self) -> None:
        self.assertEqual(serve_colab.dead_child({"vllm": ALIVE, "gate": lambda: False}), "gate")
        self.assertIsNone(serve_colab.dead_child({"vllm": ALIVE, "gate": ALIVE}))


class ProbePublicTests(unittest.TestCase):
    def test_retries_until_success(self) -> None:
        probe = Mock(side_effect=[RuntimeError("HTTP 530"), RuntimeError("HTTP 530"), None])
        with patch.object(serve_colab.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
            serve_colab.probe_public(probe, URL)
        self.assertEqual(probe.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_raises_after_last_attempt(self) -> None:
        probe = Mock(side_effect=RuntimeError("HTTP 530"))
        with patch.object(serve_colab.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                serve_colab.probe_public(probe, URL)
        self.assertEqual(probe.call_count, 6)
        self.assertEqual(sleep.call_count, 5)


class CloudflaredTests(unittest.TestCase):
    def start(self, body: str, timeout_s: float):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CLOUDFLARED_BIN": fake_cloudflared(Path(tmp), body)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    return serve_colab.start_cloudflared(8001, Path(tmp) / "cf.log", timeout_s=timeout_s)

    def test_noisy_cloudflared_never_blocks(self) -> None:
        proc, url = self.start(NOISY, 10)
        self.assertEqual(url, URL)
        self.assertEqual(proc.wait(timeout=10), 0)

    def test_exits_fast_with_log_tail_when_cloudflared_dies(self) -> None:
        t0 = time.monotonic()
        with self.assertRaises(SystemExit) as ctx:
            self.start("sys.stderr.write('ERR boom\\n'); sys.exit(1)", 30)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertIn("ERR boom", str(ctx.exception))

    def test_deadline_is_enforced_when_no_url(self) -> None:
        t0 = time.monotonic()
        with self.assertRaises(SystemExit):
            self.start("time.sleep(60)", 2)
        self.assertLess(time.monotonic() - t0, 10)

    def test_url_written_just_before_exit_is_still_found(self) -> None:
        proc, url = self.start("sys.stderr.write('INF https://abc-def.trycloudflare.com\\n')", 10)
        self.assertEqual(url, URL)
        proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
