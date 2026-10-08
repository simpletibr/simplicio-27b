"""Contract test for issue #31: everything the notebooks and scripts download is pinned.

Static checks over the tracked files, no network and no GPU:
  * every `pip install` names an exact version (`name==x.y.z`) or a `git+URL@<tag|sha>`;
  * every `git clone` names a 40-hex commit on the same line;
  * every `curl`/`wget` is followed by a sha256 check of a literal 64-hex digest;
  * no `releases/latest`, no raw.githubusercontent.com URL of a branch, no HF token placeholder;
  * the Serve notebook fetches deploy/* from SIMPLICIO_SHA, a commit of this repo that holds every
    file it downloads;
  * the Training and Merge notebooks install the same nine pins, which are the ones that resolve
    together for Colab (`uv pip compile`, issue #31).
What is NOT pinned yet is listed in OWNER_PENDING with the reason; each entry must still be true,
so a fixed item has to leave the list.
"""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE_NB = "notebooks/Simplicio_27B_Serve_Colab.ipynb"
TRAIN_NB = "notebooks/Simplicio_27B_Training_Colab.ipynb"
MERGE_NB = "notebooks/Simplicio_27B_Merge_Colab.ipynb"

VLLM_PIN = "vllm==0.31.0"
CLOUDFLARED_VERSION = "2026.10.0"
# sha256sum of cloudflared-linux-amd64 2026.10.0, computed from the downloaded release asset.
CLOUDFLARED_SHA256 = "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db"
COLAB_PINS = (
    "unsloth==2026.9.14", "unsloth_zoo==2026.9.9", "transformers==5.5.0", "trl==0.24.0",
    "peft==0.21.0", "accelerate==1.15.0", "bitsandbytes==0.50.2", "datasets==4.3.0",
    "huggingface_hub==1.33.0",
)
SCANNED_SUFFIXES = (".py", ".sh", ".ipynb", ".md", ".txt")
SCANNED_NAMES = ("Makefile", "Modelfile", "pre-push")

# Not pinned here on purpose. Key: (file, text that must still be in the file). Value: why.
OWNER_PENDING = {
    ("README.md", "git clone https://github.com/simpletibr/simplicio-27b"):
        "instruction for readers in the README; issue #31 leaves it (README belongs to #12)",
    ("notebooks/Simplicio_27B_Training_Colab.ipynb", 'model_name = "Qwen/Qwen3.8-27B"'):
        "base model read from the HF branch main, no revision=; weights review is #23/#29",
    ("notebooks/Simplicio_27B_Merge_Colab.ipynb", 'model_name="wesleysimplicio/Simplicio-27B"'):
        "adapter read from the HF branch main, no revision=; weights review is #23/#29",
    ("deploy/serve_vllm.sh", 'MODEL_ID="${1:-wesleysimplicio/Simplicio-27B}"'):
        "merged weights read from the HF branch main, no --revision; weights review is #23/#29",
    ("README.md", 'model_name="wesleysimplicio/Simplicio-27B"'):
        "adapter read from the HF branch main in the README snippet; weights review is #23/#29",
    ("deploy/serve_colab.py", 'MODEL_ID = os.environ.get("MODEL_ID", "wesleysimplicio/Simplicio-27B")'):
        "merged weights read from the HF branch main, no revision; weights review is #23/#29",
    ("README.md", "ollama run wesleysimplicio/simplicio-27b"):
        "Ollama tag `latest` without a digest; Modelfile and GGUF identity are #22/#24",
    ("Modelfile", "FROM ./Qwen3.8-27B.Q4_K_M.gguf"):
        "GGUF downloaded by hand from HF, no checksum published; #24",
}

NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
PINNED = re.compile(rf"^{NAME}==[A-Za-z0-9.+!_-]+$")
GIT_REF = re.compile(rf"^{NAME}@git\+https://[^@\s]+@(?P<ref>[A-Za-z0-9._/-]+)(#\S*)?$")
VALUE_OPTS = {"-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-i", "--index-url",
              "--extra-index-url", "-f", "--find-links", "-t", "--target"}
STOP = {"&&", "||", ";", "|", "\\", "#"}
VERIFY = re.compile(r"\b(sha256sum|shasum)\b.*(\s-c\b|\s--check\b)")
COMMAND_FETCH = re.compile(r"(?:^|&&|\|\||;|\||\()\s*[!$]?\s*(?:sudo\s+)?(?:curl|wget)\s")
HEX40 = re.compile(r"\b[0-9a-f]{40}\b")
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def tracked() -> list[str]:
    out = git("ls-files")
    assert out.returncode == 0, out.stderr
    return [p for p in out.stdout.splitlines()
            if (p.endswith(SCANNED_SUFFIXES) or Path(p).name in SCANNED_NAMES)
            and not p.startswith(("tests/", "data/")) and (ROOT / p).is_file()]


def code_chunks(rel: str) -> list[tuple[str, str]]:
    """(label, text) per code cell for notebooks; the whole file otherwise. Markdown cells are prose."""
    text = (ROOT / rel).read_text(encoding="utf-8")
    if not rel.endswith(".ipynb"):
        return [(rel, text)]
    cells = json.loads(text)["cells"]
    return [(f"{rel} cell {i}", "".join(c["source"])) for i, c in enumerate(cells)
            if c["cell_type"] == "code"]


def logical_lines(text: str) -> list[tuple[int, str]]:
    """Lines with backslash continuations joined; the number is that of the first physical line."""
    out, buf, start = [], "", 0
    for n, line in enumerate(text.splitlines(), 1):
        if not buf:
            start = n
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        out.append((start, buf + line))
        buf = ""
    if buf:
        out.append((start, buf))
    return out


def is_pinned(req: str) -> bool:
    name = re.match(r"[A-Za-z0-9._-]+", req)
    if name and name.group(0).lower() == "vllm":
        return req == VLLM_PIN
    git_ref = GIT_REF.match(req)
    return bool(PINNED.match(req)) or (git_ref is not None
                                       and git_ref["ref"] not in {"main", "master", "HEAD"})


def pip_problems(label: str, text: str) -> list[str]:
    problems = []
    for n, line in logical_lines(text):
        for m in re.finditer(r"\bpip3? install\b", line):
            tokens = iter(re.sub(r"\s+@\s+", "@", line[m.end():]).split())
            for tok in tokens:
                if tok in STOP:
                    break
                if tok in VALUE_OPTS:
                    next(tokens, None)
                elif not tok.startswith("-") and not is_pinned(tok.strip("'\"`(),")):
                    problems.append(f"{label} line {n}: {tok!r} without an exact version")
                if tok.endswith("`"):
                    break
    return problems


def fetch_problems(label: str, text: str) -> list[str]:
    problems = []
    for n, line in logical_lines(text):
        if COMMAND_FETCH.search(line) and not (VERIFY.search(line) and HEX64.search(line)):
            problems.append(f"{label} line {n}: curl/wget without `sha256sum -c` of a 64-hex digest")
    return problems


def clone_problems(label: str, text: str) -> list[str]:
    return [f"{label} line {n}: git clone without a 40-hex commit"
            for n, line in logical_lines(text) if "git clone" in line and not HEX40.search(line)]


def without_owner_pending(label_file: str, problems: list[str], text: str) -> list[str]:
    """Drop problems on lines that OWNER_PENDING covers for this file."""
    allowed = [snippet for (path, snippet) in OWNER_PENDING if path == label_file]
    if not allowed:
        return problems
    kept = []
    for p in problems:
        n = int(re.search(r" line (\d+):", p)[1])
        line = next((ln for i, ln in logical_lines(text) if i == n), "")
        if not any(snippet in line for snippet in allowed):
            kept.append(p)
    return kept


def nb_code(rel: str) -> str:
    return "\n".join(text for _, text in code_chunks(rel))


class ScannerRulesTest(unittest.TestCase):
    """The scanners themselves: they must catch the unpinned forms and accept the pinned ones."""

    def test_pip_scanner(self) -> None:
        bad = ["!pip install -q vllm", "pip install vllm==0.30.0", "pip install --no-deps 'trl<0.9.0'",
               "!pip install -q unsloth",
               'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
               'pip install "x @ git+https://github.com/o/r@main"']
        good = ['!pip install -q "vllm==0.31.0" "unsloth==2026.9.14"', "Run `pip install vllm==0.31.0` first",
                'pip install "gguf @ git+https://github.com/ggml-org/llama.cpp@v0.6.0#subdirectory=gguf-py"',
                "pip install -r requirements.txt && make", "pip install \\\n  peft==0.21.0"]
        for line in bad:
            self.assertTrue(pip_problems("x", line), line)
        self.assertEqual([p for line in good for p in pip_problems("x", line)], [])

    def test_fetch_scanner(self) -> None:
        digest = "a" * 64
        bad = ["!wget -q -O x https://h/x", "curl -fsSL https://h/install.sh | sh",
               f"wget -O x https://h/x && echo {digest} | sha256sum", "wget https://h/x && sha256sum -c SUMS"]
        good = [f'!wget -q -O x https://h/x && echo "{digest}  x" | sha256sum -c - && chmod +x x',
                f'curl -fsSLo x https://h/x \\\n && echo "{digest}  x" | shasum -a 256 -c -',
                "see the curl documentation"]
        for line in bad:
            self.assertTrue(fetch_problems("x", line), line)
        self.assertEqual([p for line in good for p in fetch_problems("x", line)], [])

    def test_clone_scanner(self) -> None:
        sha = "b" * 40
        self.assertTrue(clone_problems("x", "git clone https://h/r"))
        self.assertTrue(clone_problems("x", f"git clone https://h/r\ngit -C r checkout {sha}"))
        self.assertEqual(clone_problems("x", f"git clone https://h/r r && git -C r checkout -q {sha}"), [])


class PinnedDownloadsTest(unittest.TestCase):
    def test_pip_installs_are_pinned(self) -> None:
        problems = []
        for rel in tracked():
            for label, text in code_chunks(rel):
                problems += pip_problems(label, text)
            if Path(rel).name.startswith("requirements") and rel.endswith(".txt"):
                for n, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
                    req = re.sub(r"\s+@\s+", "@", re.sub(r"\s+#.*$", "", line).strip())
                    if req and not req.startswith(("#", "-r ", "-c ")) and not is_pinned(req):
                        problems.append(f"{rel} line {n}: {req!r} without an exact version")
        self.assertEqual(problems, [])

    def test_git_clones_name_a_commit(self) -> None:
        problems = []
        for rel in tracked():
            for label, text in code_chunks(rel):
                problems += without_owner_pending(rel, clone_problems(label, text), text)
        self.assertEqual(problems, [])

    def test_curl_and_wget_verify_a_sha256(self) -> None:
        problems = []
        for rel in tracked():
            for label, text in code_chunks(rel):
                problems += without_owner_pending(rel, fetch_problems(label, text), text)
        self.assertEqual(problems, [])

    def test_no_mutable_download_or_token_placeholder(self) -> None:
        problems = []
        for rel in tracked():
            if rel.endswith(".md"):
                continue  # prose; the logo <img> on raw.githubusercontent.com/.../main is an image
            for label, text in code_chunks(rel):
                for n, line in enumerate(text.splitlines(), 1):
                    if "releases/latest" in line:
                        problems.append(f"{label} line {n}: releases/latest")
                    if re.search(r"raw\.githubusercontent\.com/\S*/(main|master|refs/heads)/", line):
                        problems.append(f"{label} line {n}: raw.githubusercontent.com of a branch")
                    for bad in ("YOUR_HF_TOKEN_HERE", "write_permission", "unslothai/unsloth.git"):
                        if bad in line:
                            problems.append(f"{label} line {n}: {bad}")
        self.assertEqual(problems, [])

    def test_hf_model_ids_without_revision_are_listed(self) -> None:
        """A new model id read from the HF branch main has to be pinned (revision) or listed here."""
        pattern = re.compile(r'model_name\s*=\s*"[^"]+"|MODEL_ID="[^"]+"'
                             r'|MODEL_ID\s*=\s*os\.environ\.get\("MODEL_ID",\s*"[^"]+"\)')
        unlisted = []
        for rel in tracked():
            for label, text in code_chunks(rel):
                for m in pattern.finditer(text):
                    if "revision" not in text and (rel, m.group(0)) not in OWNER_PENDING:
                        unlisted.append(f"{label}: {m.group(0)}")
        self.assertEqual(unlisted, [])

    def test_modelfile_from_is_local_or_digest(self) -> None:
        rel = "Modelfile"
        froms = [ln.split(None, 1)[1].strip() for ln in (ROOT / rel).read_text(encoding="utf-8").splitlines()
                 if ln.startswith("FROM ")]
        self.assertTrue(froms, rel)
        for ref in froms:
            self.assertTrue(ref.startswith("./") or "@sha256:" in ref, f"{rel}: FROM {ref}")

    def test_owner_pending_entries_are_still_true(self) -> None:
        """A fixed item must leave OWNER_PENDING: every entry still has to be in its file."""
        for (rel, snippet), reason in OWNER_PENDING.items():
            self.assertTrue(reason)
            self.assertIn(snippet, nb_code(rel), f"stale entry: {rel}")


class ServeNotebookTest(unittest.TestCase):
    def test_cloudflared_is_a_pinned_release_with_sha256(self) -> None:
        """In the notebook now; #20 moves it to deploy/serve_colab.py with the same pin."""
        holders = [rel for rel in ("deploy/serve_colab.py", SERVE_NB)
                   if "cloudflare/cloudflared/releases/download" in nb_code(rel)]
        self.assertTrue(holders, "no pinned cloudflared download in the notebook or deploy/serve_colab.py")
        for rel in holders:
            code = nb_code(rel)
            self.assertIn(f"cloudflared/releases/download/{CLOUDFLARED_VERSION}/cloudflared-linux-amd64", code)
            self.assertIn(CLOUDFLARED_SHA256, code)
            self.assertNotIn("releases/latest", code)

    def test_deploy_files_come_from_a_commit_of_main(self) -> None:
        code = nb_code(SERVE_NB)
        found = re.search(r'SIMPLICIO_SHA = "([0-9a-f]{40})"', code)
        self.assertIsNotNone(found, "SIMPLICIO_SHA with 40 hex is missing")
        sha = found.group(1)
        self.assertIn("raw.githubusercontent.com/simpletibr/simplicio-27b/{SIMPLICIO_SHA}", code)
        self.assertEqual(git("cat-file", "-t", sha).stdout.strip(), "commit", f"{sha} is not in this clone")
        self.assertEqual(git("merge-base", "--is-ancestor", sha, "HEAD").returncode, 0,
                         f"{sha} is not an ancestor of HEAD")
        names = re.findall(r'"([^"]+)"', re.search(r"for name in \(([^)]*)\)", code)[1])
        self.assertTrue(names)
        for name in names:
            res = git("cat-file", "-e", f"{sha}:deploy/{name}")
            self.assertEqual(res.returncode, 0, f"deploy/{name} does not exist in {sha}")


class PinConsistencyTest(unittest.TestCase):
    def test_vllm_pin_is_the_same_everywhere(self) -> None:
        for rel in (SERVE_NB, "README.md", "deploy/DISTRIBUTION_GUIDE.md", "deploy/serve_vllm.sh"):
            self.assertIn(VLLM_PIN, (ROOT / rel).read_text(encoding="utf-8"), rel)
        self.assertIn(f'PINNED = "{VLLM_PIN.split("==")[1]}"',
                      (ROOT / "scripts/check_vllm_args.py").read_text(encoding="utf-8"))
        other = []
        for rel in tracked():
            for label, text in code_chunks(rel):
                other += [f"{label}: {v}" for v in re.findall(r"\bvllm==\d+(?:\.\d+)*", text) if v != VLLM_PIN]
        self.assertEqual(other, [])

    def test_training_and_merge_install_the_nine_colab_pins(self) -> None:
        for rel in (TRAIN_NB, MERGE_NB):
            code = nb_code(rel)
            installs = [ln for ln in code.splitlines() if ln.startswith("!pip install")]
            self.assertEqual(len(installs), 1, f"{rel}: one pip install, got {installs}")
            names = [t.strip('"') for t in installs[0].split() if not t.startswith(("-", "!"))
                     and t != "install"]
            self.assertEqual(names, list(COLAB_PINS), rel)
            self.assertIn("!pip list --format=freeze", code, f"{rel}: installed versions are not logged")


if __name__ == "__main__":
    unittest.main()
