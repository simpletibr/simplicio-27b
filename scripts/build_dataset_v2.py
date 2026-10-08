"""Build data/v2: real small Python commits as SEARCH/REPLACE rows in the serving message format (#28).

Every row is a real edit: `messages` = system + user + assistant, where the user message is the
`USER_TEMPLATE` of benchmarks/harness/harness.py filled with the parent version of the file and the
instruction, and the assistant message is only the SEARCH/REPLACE blocks (markers of 4 characters)
that turn the parent file into the committed file, byte for byte. Nothing is synthetic: a commit
that cannot be turned into such a row is dropped, never invented. The `scope` and
`instruction_source` fields say how faithful the instruction is to the diff.

Sources:
  * the history of this repository, read up to a pinned commit (default, needs only git);
  * `--external`: the public repositories listed in EXTERNAL (issue #28), cloned bare into --workdir.
    They are not fetched unless asked for.

Decontamination uses the criterion of benchmarks/harness/decontam_tasks.py (13-grams of
`\\w+` tokens, every string of the row against the task text) against benchmarks/harness/tasks/
and, in the same way, against the old eval file data/unseen_eval_120.json.

Usage: python3 scripts/build_dataset_v2.py [--external] [--limit N] [--out DIR] [--min-train N]
"""

from __future__ import annotations

import argparse
import ast
import collections
import difflib
import importlib.util
import json
import random
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = ROOT / "benchmarks" / "harness"


def _harness_module(name: str):
    """Load benchmarks/harness/<name>.py without touching the import path; a module already imported under that name is reused."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Training and eval share one prompt and one SEARCH/REPLACE parser: the ones of the harness (#25).
harness = _harness_module("harness")
decontam_tasks = _harness_module("decontam_tasks")
SYSTEM_PROMPT, USER_TEMPLATE, FORMAT_HINT = harness.SYSTEM_PROMPT, harness.USER_TEMPLATE, harness.FORMAT_HINT
PatchError, apply_blocks, parse_blocks = harness.PatchError, harness.apply_blocks, harness.parse_blocks

SEED = 27
PER_REPO_CAP = 300
MIN_TRAIN = 2000
VAL_REPO = "python-poetry/poetry"
MAX_CHANGED_LINES = 60
MAX_FILE_LINES = 400
MAX_FILE_CHARS = 16000
EVAL_FILE = ROOT / "data" / "unseen_eval_120.json"
OUT_DIR = ROOT / "data" / "v2"
BANNED = ("COMMIT_READY", "VERIFIED_GREEN", "<validate>", "<deliver>", "simplicio_deliver")


@dataclass(frozen=True)
class Source:
    name: str
    license: str
    pin: str  # every ancestor of this commit is read, so the result does not depend on when the clone was made
    license_file: str
    url: str = ""  # "" = the repository this script lives in; otherwise cloned bare into --workdir
    require_at_pin: bool = False  # drop edits of files that do not exist at `pin` any more
    curated: dict = field(default_factory=dict)  # commit sha -> instruction vetted against the diff


# Instructions written for commits whose message is not an imperative English sentence or that touch
# several files. Each one was checked against `git show <sha> -- <path>`; it describes only that file.
CURATED = {
    "c42b560fe8200a179c55fa9ccb50a76e1aaad83d": (
        "Envie a chave admin do gateway no header Authorization: Bearer, nunca na query string da URL, "
        "em set_upstream e clear_upstream, fazendo _http_json aceitar headers extras. "
        "Leia a chave com getpass quando SIMPLETI_ADMIN_KEY nao estiver definida."
    ),
}
# This repository: the history up to the main that existed when the dataset was built. The repo itself
# removed many of its old Python files (142e9f6 "simulated, withdrawn or obsolete numbers", d234d83
# "one-shot generators"), so edits of files that are gone at the pin do not become training data.
LOCAL = Source(
    "simpletibr/simplicio-27b", "Apache-2.0", "b61dccd400457c5112a2c2c1c74b94a90edc8376", "LICENSE",
    require_at_pin=True, curated=CURATED,
)
# (repo, SPDX license from the GitHub API, pinned end commit, license file) as given in issue #28.
# Pins and licenses were not checked against the network when this list was added.
_EXTERNAL = [
    ("django/django", "BSD-3-Clause", "135d7c606fc66a9f47f690e5f242ced386524954", "LICENSE"),
    ("ipython/ipython", "BSD-3-Clause", "93dde62b5b76275dbd1a1cfda328969fa59caa08", "LICENSE"),
    ("scrapy/scrapy", "BSD-3-Clause", "eba114d98ecd3986f23af6facb524fc4ac4d8f71", "LICENSE"),
    ("pypa/pip", "MIT", "a7002c9771a6c3f0317a4e6b9fbdcd22e643f7b6", "LICENSE.txt"),
    ("conan-io/conan", "MIT", "e85a86e84b75a8bc80ba17e19be8df7e5c6df1fa", "LICENSE.md"),
    ("mitmproxy/mitmproxy", "MIT", "5253dcbd1d8f0522de097bfe56918fe12a0f267a", "LICENSE"),
    ("dask/dask", "BSD-3-Clause", "6ce3740fa79bfa051ecc74c3641f8c6a46dfdc30", "LICENSE.txt"),
    ("pytest-dev/pytest", "MIT", "59200df778adf8079df6125d60662908ac0a0646", "LICENSE"),
    ("pallets/werkzeug", "BSD-3-Clause", "594452f6a4fe4de38a544962fbf04bfc9d37fbc2", "LICENSE.txt"),
    ("python-poetry/poetry", "MIT", "63bba274f8ea6bc9a064f5328fa6c67bce617cc1", "LICENSE"),
]
EXTERNAL = [Source(r, lic, pin, lf, url=f"https://github.com/{r}.git") for r, lic, pin, lf in _EXTERNAL]

SKIP_PATH_RE = re.compile(
    r"(^|/)(tests?|testing|docs?|examples?|_vendor|vendor|extern|external|externals|third_party)/"
    r"|(^|/)(test_[^/]*|[^/]*_test|conftest|setup)\.py$"
    r"|^benchmarks/harness/tasks/|^data/"
)
PREFIX_RE = re.compile(r"^(fix|feat|refactor|perf|docs|chore|style|tests?|build|ci)(\([^)]*\))?!?:\s*", re.I)
PR_RE = re.compile(r"\s*\((?:#|gh-)\d+\)\s*$")
TRAILER_RE = re.compile(r"^(signed-off-by|co-authored-by|reviewed-by|fixes|closes|refs|see):", re.I)
VERBS = (
    "add allow avoid catch change check clean convert correct deprecate disable drop enable ensure "
    "expose fix handle ignore implement improve introduce keep make move normalize pass prevent raise "
    "reduce refactor remove rename replace respect restore return simplify skip support switch update "
    "use validate warn"
).split()
IRREGULAR = {"made": "make", "caught": "catch", "kept": "keep", "dropped": "drop", "skipped": "skip",
             "simplified": "simplify"}
# A literal credential in the code or in the diff must never become training data.
SECRET_RE = re.compile(
    r"(?i)[?&](?:api_?)?key=[A-Za-z0-9_\-]{12,}"
    r"|(?:api[_-]?key|secret|token|passw(?:or)?d)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"
    r"|\b(?:sk|hf|ghp|gho|xox[bap])[-_][A-Za-z0-9_\-]{16,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def git(repo_dir: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, check=True).stdout


def clone(source: Source, workdir: Path) -> Path:
    repo_dir = workdir / (source.name.replace("/", "__") + ".git")
    if not repo_dir.exists():
        subprocess.run(["git", "clone", "--bare", "--quiet", source.url, str(repo_dir)], check=True)
    if subprocess.run(["git", "-C", str(repo_dir), "cat-file", "-e", f"{source.pin}^{{commit}}"]).returncode:
        git(repo_dir, "fetch", "--quiet", "origin")
        git(repo_dir, "cat-file", "-e", f"{source.pin}^{{commit}}")
    return repo_dir


def candidates(repo_dir: Path, pin: str) -> tuple[list[tuple[str, str, str, int, int]], int]:
    """((commit, parent, path, changed_lines, files_in_commit), ...) for small edits of one .py file, and the commit count."""
    out, cur, total = [], None, 0
    log = git(repo_dir, "log", "--no-merges", "--numstat", "--format=@@%H %P", pin).decode("utf-8", "replace")
    for line in log.splitlines() + ["@@"]:
        if line.startswith("@@"):
            if cur and len(cur[1]) == 1:
                for added, deleted, path in cur[2]:
                    if (path.endswith(".py") and "=>" not in path and added.isdigit() and deleted.isdigit()
                            and not SKIP_PATH_RE.search(path)
                            and 1 <= int(added) + int(deleted) <= MAX_CHANGED_LINES):
                        out.append((cur[0], cur[1][0], path, int(added) + int(deleted), len(cur[2])))
            parts = line[2:].split()
            cur = (parts[0], parts[1:], []) if parts else None
            total += 1 if parts else 0
        elif line.strip() and cur:
            cur[2].append(tuple(line.split("\t", 2)))
    return out, total


def imperative(word: str) -> str | None:
    w = word.lower()
    if w in IRREGULAR:
        return IRREGULAR[w]
    for v in VERBS:
        if w in (v, v + "s", v + "es", v + "ed", v + "d", v + "ing", v[:-1] + "ing"):
            return v
    return None


def instruction_from_commit(message: str) -> str | None:
    if not message.strip():
        return None
    paras = re.split(r"\n\s*\n", message.strip())
    subject = PR_RE.sub("", PREFIX_RE.sub("", paras[0].splitlines()[0].strip())).strip().rstrip(".")
    words = subject.split()
    if not 15 <= len(subject) <= 120 or len(words) < 3:
        return None
    verb = imperative(words[0])
    if verb is None:
        return None
    text = " ".join([verb.capitalize(), *words[1:]]) + "."
    if len(paras) > 1:
        body = " ".join(x.strip() for x in paras[1].splitlines() if x.strip() and not TRAILER_RE.match(x.strip()))
        if 20 <= len(body) <= 500 and "http" not in body:
            text += " " + body
    return text


def code_dump(src: str) -> str:
    """ast.dump without docstrings, so comment- or docstring-only commits compare equal."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def blocks_for(old: str, new: str) -> list[tuple[str, str]]:
    a, b = old.splitlines(), new.splitlines()
    return [("\n".join(a[g[0][1]:g[-1][2]]), "\n".join(b[g[0][3]:g[-1][4]]))
            for g in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(3)]


def render(blocks: list[tuple[str, str]]) -> str:
    return "\n".join(f"<<<< SEARCH\n{s}\n====\n{r}\n>>>> REPLACE" for s, r in blocks)


def commit_messages(repo_dir: Path, pin: str) -> dict[str, str]:
    log = git(repo_dir, "log", "--no-merges", "--format=%H%x00%B%x1e", pin).decode("utf-8", "replace")
    return dict(chunk.strip("\n").split("\0", 1) for chunk in log.split("\x1e") if "\0" in chunk)


class Blobs:
    """One `git cat-file --batch` process per repo: read(rev, path) -> file text, or None."""

    def __init__(self, repo_dir: Path) -> None:
        self.proc = subprocess.Popen(["git", "-C", str(repo_dir), "cat-file", "--batch"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def read(self, rev: str, path: str) -> str | None:
        self.proc.stdin.write(f"{rev}:{path}\n".encode("utf-8"))
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().split()
        if len(header) != 3:
            return None
        data = self.proc.stdout.read(int(header[2]))
        self.proc.stdout.read(1)
        try:
            return data.decode("utf-8") if header[1] == b"blob" else None
        except UnicodeDecodeError:
            return None

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        self.proc.stdout.close()


def make_row(source: Source, blobs: Blobs, message: str, cand: tuple[str, str, str, int, int]) -> tuple[dict | None, str]:
    commit, parent, path, changed, n_files = cand
    if commit in source.curated:
        instruction, instruction_source = source.curated[commit], "curated_from_commit"
    elif n_files == 1:
        instruction, instruction_source = instruction_from_commit(message), "commit_subject"
        if instruction is None:
            return None, "bad_message"
    else:
        # The subject describes the whole commit, so it is not a faithful instruction for one of its files.
        return None, "multi_file_commit"
    old, new = blobs.read(parent, path), blobs.read(commit, path)
    if old is None or new is None:
        return None, "not_text_modification"
    if source.require_at_pin and blobs.read(source.pin, path) is None:
        return None, "removed_from_pin"
    if "\r" in old or "\r" in new:
        return None, "crlf"
    if len(old.splitlines()) > MAX_FILE_LINES or len(old) > MAX_FILE_CHARS:
        return None, "file_too_long"
    try:
        if code_dump(old) == code_dump(new):
            return None, "no_code_change"
    except (SyntaxError, ValueError):
        return None, "syntax_error"
    answer = render(blocks_for(old, new))
    try:
        if apply_blocks(old, parse_blocks(answer)) != new:
            return None, "roundtrip_mismatch"
    except PatchError:
        return None, "roundtrip_mismatch"
    if any(b in answer for b in BANNED):
        return None, "banned_marker"
    if any(SECRET_RE.search(text) for text in (old, new, instruction)):
        return None, "secret"
    user = USER_TEMPLATE.format(context=f"Projeto Python {source.name}.", edit_file=path, code=old,
                                instruction=instruction)
    scope = "single_file_commit" if n_files == 1 else "multi_file_commit"
    return {"id": f"{source.name}@{commit[:12]}:{path}", "repo": source.name, "license": source.license,
            "commit": commit, "parent": parent, "path": path, "changed_lines": changed,
            "source": "real_commit", "scope": scope, "instruction_source": instruction_source,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}, "kept"


def eval_grams(tasks_dir: Path = decontam_tasks.TASKS, eval_file: Path = EVAL_FILE) -> set[tuple[str, ...]]:
    """13-grams of every eval task: instruction + code + oracle (as decontam_tasks does) and every old eval row."""
    grams: set[tuple[str, ...]] = set()
    for d in decontam_tasks.task_dirs(tasks_dir):
        grams |= decontam_tasks.ngrams(decontam_tasks.task_text(d))
    if eval_file.is_file():
        for row in json.loads(eval_file.read_text(encoding="utf-8")):
            grams |= decontam_tasks.ngrams("\n".join(decontam_tasks.strings(row)))
    if not grams:
        raise SystemExit(f"no eval text found under {tasks_dir}")
    return grams


def contaminated(row: dict, grams: set[tuple[str, ...]]) -> bool:
    """Same test as decontam_tasks.build_report: any string of the row shares a 13-gram with the eval."""
    return any(not decontam_tasks.ngrams(text).isdisjoint(grams) for text in decontam_tasks.strings(row))


def process(source: Source, repo_dir: Path, grams: set, cap: int, limit: int | None) -> tuple[list[dict], dict]:
    cands, total = candidates(repo_dir, source.pin)
    messages, blobs = commit_messages(repo_dir, source.pin), Blobs(repo_dir)
    rows, reasons = [], collections.Counter()
    try:
        for cand in cands:
            row, reason = make_row(source, blobs, messages.get(cand[0], ""), cand)
            reasons[reason] += 1
            if row is not None:
                rows.append(row)
    finally:
        blobs.close()
    clean = [r for r in rows if not contaminated(r, grams)]
    reasons["contaminated"] = len(rows) - len(clean)
    reasons["kept"] -= len(rows) - len(clean)
    clean.sort(key=lambda r: (r["commit"], r["path"]))
    used = clean if len(clean) <= cap else random.Random(f"{SEED}:{source.name}").sample(clean, cap)
    if limit is not None:
        used = used[:limit]
    stats = {"commits": total, "candidates": len(cands), "validated": len(rows),
             "contaminated": len(rows) - len(clean), "used": len(used),
             "reasons": {k: v for k, v in sorted(reasons.items()) if v}}
    return used, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_stats(out: Path, per_source: dict, train: list, val: list, pending: list[Source], processed: list[Source]) -> None:
    sizes = [sum(len(m["content"]) for m in r["messages"]) for r in train + val] or [0]
    lines = [
        "# Dataset v2: estatisticas", "",
        "Gerado por `python3 scripts/build_dataset_v2.py`. Nao edite a mao.", "",
        f"- Seed: {SEED}", f"- Teto por repo: {PER_REPO_CAP}", f"- Repo de validacao: {VAL_REPO}",
        f"- Filtros: commit sem merge, 1 a {MAX_CHANGED_LINES} linhas alteradas em um arquivo .py fora de testes, docs, "
        f"vendor, benchmarks/harness/tasks e data; arquivo pai <= {MAX_FILE_LINES} linhas e <= {MAX_FILE_CHARS} caracteres; "
        "mudanca de codigo (AST sem docstrings); round-trip SEARCH/REPLACE exato; sem credencial literal; "
        "13-gram contra benchmarks/harness/tasks (instrucao, codigo e oracle) e contra data/unseen_eval_120.json "
        "(criterio de benchmarks/harness/decontam_tasks.py)",
        "- Instrucao: assunto do commit quando ele toca 1 arquivo e comeca com um verbo no imperativo; "
        "texto vetado contra o diff (`curated_from_commit`) nos outros casos",
        f"- Linhas: train {len(train)}, val {len(val)}",
        f"- Origem: {len(train) + len(val)} de commit real, 0 sinteticos",
        f"- Marco de treino: {MIN_TRAIN} linhas em train; faltam {max(0, MIN_TRAIN - len(train))}",
        f"- Caracteres por linha: max {max(sizes)}, media {sum(sizes) // len(sizes)}", "",
        "Texto de cada licenca: `data/v2/LICENSES/<owner>__<repo>.txt`.", "",
        "| Repo | Licenca | Commit fixado | Commits | Candidatos | Validados | Contaminados | Usados | Split |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in processed:
        st = per_source[s.name]
        lines.append(f"| {s.name} | {s.license} | `{s.pin}` | {st['commits']} | {st['candidates']} | {st['validated']} | "
                     f"{st['contaminated']} | {st['used']} | {'val' if s.name == VAL_REPO else 'train'} |")
    lines += ["", "Descartes por motivo (candidatos que nao viraram linha):", ""]
    for s in processed:
        reasons = {k: v for k, v in per_source[s.name]["reasons"].items() if k != "kept"}
        lines.append(f"- {s.name}: " + (", ".join(f"{k} {v}" for k, v in reasons.items()) or "nenhum"))
    if pending:
        lines += ["", "## Pendencias", "",
                  "Repositorios nao processados nesta geracao (rode `python3 scripts/build_dataset_v2.py --external --min-train "
                  f"{MIN_TRAIN}`). Pins e licencas vem da issue #28 e nao foram conferidos na rede.", "",
                  "| Repo | Licenca | Commit fixado | Arquivo de licenca | Split |", "|---|---|---|---|---|"]
        lines += [f"| {s.name} | {s.license} | `{s.pin}` | {s.license_file} | {'val' if s.name == VAL_REPO else 'train'} |"
                  for s in pending]
    (out / "STATS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(sources: list[Source], out: Path, workdir: Path, grams: set, limit: int | None = None,
          cap: int = PER_REPO_CAP, local_repo: Path = ROOT, pending: list[Source] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    (out / "LICENSES").mkdir(exist_ok=True)
    train, val, per_source = [], [], {}
    for source in sources:
        repo_dir = clone(source, workdir) if source.url else local_repo
        (out / "LICENSES" / (source.name.replace("/", "__") + ".txt")).write_bytes(
            git(repo_dir, "show", f"{source.pin}:{source.license_file}"))
        used, per_source[source.name] = process(source, repo_dir, grams, cap, limit)
        print(source.name, json.dumps(per_source[source.name], sort_keys=True), flush=True)
        (val if source.name == VAL_REPO else train).extend(used)
    random.Random(SEED).shuffle(train)
    random.Random(SEED).shuffle(val)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_stats(out, per_source, train, val, pending or [], sources)
    return {"train": train, "val": val, "per_source": per_source}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--workdir", type=Path, default=Path.home() / ".cache" / "simplicio-dataset-v2")
    ap.add_argument("--tasks", type=Path, default=decontam_tasks.TASKS)
    ap.add_argument("--eval-file", type=Path, default=EVAL_FILE)
    ap.add_argument("--external", action="store_true", help="also clone and process the repositories of EXTERNAL")
    ap.add_argument("--limit", type=int, default=None, help="small mode: at most N rows per source")
    ap.add_argument("--min-train", type=int, default=0, help="exit with 1 when train has fewer rows")
    args = ap.parse_args(argv)
    sources = [LOCAL, *EXTERNAL] if args.external else [LOCAL]
    result = build(sources, args.out, args.workdir, eval_grams(args.tasks, args.eval_file), args.limit,
                   pending=[] if args.external else EXTERNAL)
    train, val = result["train"], result["val"]
    print(f"train {len(train)} val {len(val)}")
    if len(train) < args.min_train:
        print(f"FAIL: train has {len(train)} rows, milestone is {args.min_train}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
