import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOOD = "Bearer " + "a" * 64


@unittest.skipUnless(shutil.which("php"), "php não instalado")
class SetUpstreamPhpTests(unittest.TestCase):
    admin_key = "a" * 64

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.state = Path(cls.tmp.name) / "state" / "upstream.json"
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        cls.port = sock.getsockname()[1]
        sock.close()
        cls.proc = subprocess.Popen(
            ["php", "-S", f"127.0.0.1:{cls.port}", str(ROOT / "deploy" / "set_upstream.php")],
            env={
                **os.environ,
                "SIMPLETI_ADMIN_KEY": cls.admin_key,
                "SIMPLETI_UPSTREAM_FILE": str(cls.state),
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", cls.port), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    cls.proc.kill()
                    cls.tmp.cleanup()
                    raise RuntimeError("php -S não subiu")
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proc.terminate()
        cls.proc.wait(timeout=5)
        cls.tmp.cleanup()

    def call(self, method="POST", body=None, auth=None, query=""):
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = auth
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/set_upstream.php{query}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, dict(resp.headers), resp.read().decode()
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, dict(exc.headers), exc.read().decode()


class AuthAndMethodTests(SetUpstreamPhpTests):
    def test_get_is_405(self) -> None:
        status, headers, _ = self.call("GET", auth=GOOD)
        self.assertEqual(status, 405)
        self.assertEqual(headers.get("Allow"), "POST")

    def test_no_header_is_401(self) -> None:
        status, _, _ = self.call(body={"upstream_url": "https://ok.trycloudflare.com"})
        self.assertEqual(status, 401)

    def test_query_key_is_ignored(self) -> None:
        status, _, _ = self.call(
            body={"upstream_url": "https://ok.trycloudflare.com"}, query="?key=" + "a" * 64
        )
        self.assertEqual(status, 401)

    def test_array_key_no_fatal(self) -> None:
        status, _, text = self.call(body={}, query="?key[]=x")
        self.assertEqual(status, 401)
        self.assertNotIn("Fatal", text)
        self.assertNotIn(".php", text)

    def test_wrong_bearer_is_401(self) -> None:
        status, headers, _ = self.call(body={"clear": True}, auth="Bearer " + "b" * 64)
        self.assertEqual(status, 401)
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")


class UpstreamUrlTests(SetUpstreamPhpTests):
    def test_rejects_bad_urls(self) -> None:
        for url in (
            "https://x.trycloudflare.com.evil.net",
            "https://127.0.0.1:2083",
            "http://x.trycloudflare.com",
            "https://x.trycloudflare.com/",
            "https://x.trycloudflare.com\n",
            "https://attacker.example/steal?x=1",
        ):
            with self.subTest(url=url):
                status, _, _ = self.call(body={"upstream_url": url}, auth=GOOD)
                self.assertEqual(status, 400)

    def test_store_then_clear(self) -> None:
        url = "https://abc-def-ghi.trycloudflare.com"
        status, _, _ = self.call(body={"upstream_url": url}, auth=GOOD)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(self.state.read_text())["upstream_url"], url)
        self.assertEqual(sorted(p.name for p in self.state.parent.iterdir()), ["upstream.json"])
        status, _, _ = self.call(body={"clear": True}, auth=GOOD)
        self.assertEqual(status, 200)
        self.assertFalse(self.state.exists())


class ShortServerKeyTests(SetUpstreamPhpTests):
    admin_key = "short-key"

    def test_short_server_key_is_500(self) -> None:
        status, _, _ = self.call(body={"clear": True}, auth="Bearer short-key")
        self.assertEqual(status, 500)


if __name__ == "__main__":
    unittest.main()
