"""Flow test for issue #31: the pinned install and download steps still do their job, offline.

  * Serve notebook, cell "deploy files": run it with urlretrieve replaced by `git show <sha>:<path>`.
    The URLs must carry SIMPLICIO_SHA; the files it "downloads" must then build the vLLM argv
    (deploy/serve_vllm.sh with a fake `vllm`).
  * Serve notebook, cloudflared line: run it in a temp dir with a fake `wget`. A wrong sha256 or a
    failed download exits non-zero and leaves no `cloudflared` file; the right sha256 leaves an
    executable one.
  * Every `!pip install` line of the three notebooks, run with a fake `pip`, hands pip exactly the
    pinned requirements.
No network, no GPU, no real vllm, no real pip.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE_NB = "notebooks/Simplicio_27B_Serve_Colab.ipynb"
TRAIN_NB = "notebooks/Simplicio_27B_Training_Colab.ipynb"
MERGE_NB = "notebooks/Simplicio_27B_Merge_Colab.ipynb"
CLOUDFLARED_URL = ("https://github.com/cloudflare/cloudflared/releases/download/2026.10.0/"
                   "cloudflared-linux-amd64")
CLOUDFLARED_SHA256 = "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db"
COLAB_PINS = [
    "unsloth==2026.9.14", "unsloth_zoo==2026.9.9", "transformers==5.5.0", "trl==0.24.0",
    "peft==0.21.0", "accelerate==1.15.0", "bitsandbytes==0.50.2", "datasets==4.3.0",
    "huggingface_hub==1.33.0",
]
RAW_PREFIX = "https://raw.githubusercontent.com/simpletibr/simplicio-27b/"

# Replaces urllib.request.urlretrieve and then runs the notebook cell given as argv[1].
FETCH_RUNNER = r"""
import json, subprocess, sys, urllib.request
ROOT, calls = sys.argv[2], []

def fake_urlretrieve(url, dest):
    calls.append([url, dest])
    prefix = "https://raw.githubusercontent.com/simpletibr/simplicio-27b/"
    if not url.startswith(prefix):
        return  # SIMPLICIO_RAW override: nothing to read from git
    sha, _, rel = url[len(prefix):].partition("/")
    data = subprocess.run(["git", "-C", ROOT, "show", f"{sha}:{rel}"], capture_output=True, check=True).stdout
    open(dest, "wb").write(data)

urllib.request.urlretrieve = fake_urlretrieve
exec(compile(sys.argv[1], "notebook-cell", "exec"))
print(json.dumps(calls))
"""
FAKE_WGET = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_WGET_ARGS"
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-O" ]; then out="$2"; shift; fi
  shift
done
printf '%s' "$FAKE_WGET_PAYLOAD" > "$out"
[ -z "${FAKE_WGET_FAIL:-}" ] || exit 4
"""
FAKE_VLLM = "#!/bin/sh\nprintf '%s\\n' \"$@\"\n"
FAKE_PIP = "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$FAKE_PIP_LOG\"\n"


def code_cells(rel: str) -> list[str]:
    nb = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell_with(rel: str, needle: str) -> str:
    found = [c for c in code_cells(rel) if needle in c]
    if len(found) != 1:
        raise AssertionError(f"{rel}: {len(found)} code cells contain {needle!r}, expected 1")
    return found[0]


def write_exe(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in {"SIMPLICIO_RAW", "HOST", "GPU_MEM", "VLLM_BIN", "PYTHONPATH"}}
    env.update(extra)
    return env


class DeployFetchFlowTest(unittest.TestCase):
    """Notebook cell 'deploy files' -> deploy/ in the working dir -> serve_vllm.sh argv."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.cell = cell_with(SERVE_NB, "urlretrieve")

    def run_cell(self, **env: str) -> list[list[str]]:
        res = subprocess.run([sys.executable, "-I", "-c", FETCH_RUNNER, self.cell, str(ROOT)],
                             cwd=self.work, env=clean_env(**env), capture_output=True, text=True, timeout=60)
        self.assertEqual(res.returncode, 0, res.stderr)
        return json.loads(res.stdout.strip().splitlines()[-1])

    def test_downloads_come_from_the_pinned_commit(self) -> None:
        sha = re.search(r'SIMPLICIO_SHA = "([0-9a-f]{40})"', self.cell)[1]
        calls = self.run_cell()
        self.assertEqual([c[1] for c in calls],
                         [f"deploy/{n}" for n in ("context.env", "completion_gate.py", "serve_colab.py",
                                                  "serve_vllm.sh")])
        for url, dest in calls:
            self.assertEqual(url, f"{RAW_PREFIX}{sha}/{dest}")
            self.assertNotIn("/main/", url)

    def test_fetched_files_build_the_vllm_argv(self) -> None:
        self.run_cell()
        fake = self.work / "vllm"
        write_exe(fake, FAKE_VLLM)
        res = subprocess.run(["bash", "deploy/serve_vllm.sh"], cwd=self.work, capture_output=True, text=True,
                             env=clean_env(VLLM_BIN=str(fake)), timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        argv = res.stdout.splitlines()
        argv = argv[argv.index("serve"):]
        context = dict(ln.split("=", 1) for ln in (self.work / "deploy/context.env").read_text().splitlines()
                       if re.match(r"^[A-Z_]+=", ln))
        self.assertEqual(argv[0], "serve")
        self.assertEqual(argv[argv.index("--max-model-len") + 1], context["MAX_MODEL_LEN"])
        self.assertIn("simplicio-27b", argv[argv.index("--served-model-name") + 1:])
        self.assertIn("--tool-call-parser", argv)

    def test_raw_override_still_wins(self) -> None:
        calls = self.run_cell(SIMPLICIO_RAW="http://127.0.0.1:9/fork")
        self.assertTrue(calls)
        for url, dest in calls:
            self.assertEqual(url, f"http://127.0.0.1:9/fork/{dest}")


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "needs bash and sha256sum")
class CloudflaredChecksumFlowTest(unittest.TestCase):
    """The notebook line that downloads cloudflared, run with a fake wget in a temp dir."""

    @classmethod
    def setUpClass(cls) -> None:
        notebook = [ln for ln in "\n".join(code_cells(SERVE_NB)).splitlines() if "cloudflared/releases" in ln]
        if not notebook:
            raise unittest.SkipTest("cloudflared download left the notebook (#20); "
                                    "test_contract_31 still checks its pin in deploy/serve_colab.py")
        assert len(notebook) == 1, notebook
        cls.line = notebook[0].lstrip("!")

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name) / "work"
        self.bin = Path(tmp.name) / "bin"
        self.work.mkdir()
        self.bin.mkdir()
        write_exe(self.bin / "wget", FAKE_WGET)
        self.args_out = Path(tmp.name) / "wget-args.txt"

    def run_line(self, payload: bytes, line: str | None = None, fail: bool = False) -> subprocess.CompletedProcess:
        env = clean_env(PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}", FAKE_WGET_ARGS=str(self.args_out),
                        FAKE_WGET_PAYLOAD=payload.decode("latin-1"))
        if fail:
            env["FAKE_WGET_FAIL"] = "1"
        return subprocess.run(["bash", "-c", line or self.line], cwd=self.work, env=env,
                              capture_output=True, text=True, timeout=30)

    def test_downloads_the_pinned_release(self) -> None:
        self.run_line(b"anything")
        args = self.args_out.read_text().splitlines()
        self.assertIn(CLOUDFLARED_URL, args)
        self.assertEqual(args[args.index("-O") + 1], "cloudflared")
        self.assertNotIn("latest", " ".join(args))

    def test_wrong_checksum_stops_with_nonzero_and_leaves_no_binary(self) -> None:
        res = self.run_line(b"not the cloudflared release")
        self.assertNotEqual(res.returncode, 0, res.stdout)
        self.assertIn("FAILED", res.stdout + res.stderr)
        self.assertFalse((self.work / "cloudflared").exists())

    def test_failed_download_stops_with_nonzero_and_leaves_no_binary(self) -> None:
        res = self.run_line(b"half a download", fail=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse((self.work / "cloudflared").exists())

    def test_right_checksum_installs_an_executable(self) -> None:
        payload = b"#!/bin/sh\necho fake cloudflared\n"
        line = self.line.replace(CLOUDFLARED_SHA256, sha256(payload).hexdigest())
        self.assertNotEqual(line, self.line)
        res = self.run_line(payload, line)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        binary = self.work / "cloudflared"
        self.assertEqual(binary.read_bytes(), payload)
        self.assertTrue(os.access(binary, os.X_OK))


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class PipInstallFlowTest(unittest.TestCase):
    """Each `!pip install` line, run against a fake pip, passes exactly the pinned requirements."""

    def install_args(self, rel: str) -> list[list[str]]:
        lines = [ln.lstrip("!") for cell in code_cells(rel) for ln in cell.splitlines()
                 if ln.startswith("!pip install")]
        self.assertTrue(lines, rel)
        runs = []
        for line in lines:
            with tempfile.TemporaryDirectory() as tmp:
                bin_dir, log = Path(tmp) / "bin", Path(tmp) / "pip.log"
                bin_dir.mkdir()
                write_exe(bin_dir / "pip", FAKE_PIP)
                env = clean_env(PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", FAKE_PIP_LOG=str(log))
                res = subprocess.run(["bash", "-c", line], cwd=tmp, env=env, capture_output=True, text=True,
                                     timeout=30)
                self.assertEqual(res.returncode, 0, res.stderr)
                runs.append(log.read_text().splitlines())
        return runs

    def test_serve_notebook_installs_the_pinned_vllm(self) -> None:
        runs = self.install_args(SERVE_NB)
        self.assertEqual(runs, [["install", "-q", "vllm==0.31.0"]])

    def test_training_notebook_installs_the_nine_pins(self) -> None:
        self.assertEqual(self.install_args(TRAIN_NB), [["install", "--no-cache-dir", *COLAB_PINS]])

    def test_merge_notebook_installs_the_nine_pins(self) -> None:
        self.assertEqual(self.install_args(MERGE_NB), [["install", "-q", *COLAB_PINS]])

    def test_training_notebook_asks_for_the_token_instead_of_embedding_it(self) -> None:
        code = "\n".join(code_cells(TRAIN_NB))
        self.assertNotIn("YOUR_HF_TOKEN_HERE", code)
        self.assertIn('getpass("HF token (write): ")', code)
        self.assertIn('HF_TOKEN = os.environ["HF_TOKEN"]', code)


if __name__ == "__main__":
    unittest.main()
