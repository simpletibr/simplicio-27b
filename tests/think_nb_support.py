"""Helpers for the tests that read the two notebooks of decide/think.  Not a test module.

A notebook code cell is Python plus IPython lines that start with `!`: a shell command in which
`{name}` is replaced by the value of a Python variable of the notebook.  These helpers read the
notebooks as data and reproduce exactly those two rules, so a test can parse or run the command a
user would run, with the parameters a user would set.  Standard library only.
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks" / "decide"
NOTEBOOKS = {"kaggle": NB_DIR / "think_gt_kaggle.ipynb", "api": NB_DIR / "think_gt_api.ipynb"}
MANIFEST_PATH = "decide/think/gt-sample-manifest.json"          # how the notebooks name it (cwd = the checkout)
INTERPOLATION = re.compile(r"\{([^{}]*)\}")


def load(key: str) -> Dict[str, Any]:
    return json.loads(NOTEBOOKS[key].read_text(encoding="utf-8"))


def source(cell: Dict[str, Any]) -> str:
    return "".join(cell["source"])


def code_cells(key: str) -> List[str]:
    return [source(c) for c in load(key)["cells"] if c["cell_type"] == "code"]


def markdown_cells(key: str) -> List[str]:
    return [source(c) for c in load(key)["cells"] if c["cell_type"] == "markdown"]


def joined_lines(text: str) -> List[str]:
    """Physical lines with backslash continuations joined (IPython does the same for a `!` line)."""
    out, buf = [], ""
    for line in text.split("\n"):
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def shell_commands(key: str) -> List[Tuple[int, str]]:
    """(index among the code cells, command without the `!`) for every shell line."""
    found = []
    for i, text in enumerate(code_cells(key)):
        for line in joined_lines(text):
            if line.lstrip().startswith("!"):
                found.append((i, line.lstrip()[1:].strip()))
    return found


def commands_of(key: str, module: str) -> List[str]:
    """The shell commands that run `python -m <module>`."""
    return [c for _, c in shell_commands(key) if re.match(rf"python3? -m {re.escape(module)}(\s|$)", c)]


def parameters(key: str, **overrides: Any) -> Dict[str, Any]:
    """Run the first code cell (the one the user edits) and return its variables.

    `overrides` replace the value on the line `NAME = ...` (the comment after it goes with it); every name
    must be on exactly one such line, so a test cannot silently set a variable the notebook does not have.
    """
    text = code_cells(key)[0]
    for name, value in overrides.items():
        pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", re.M)
        if len(pattern.findall(text)) != 1:
            raise KeyError(f"{key}: the parameter cell has no single line `{name} = ...`")
        text = pattern.sub(lambda _m, n=name, v=value: f"{n} = {v!r}", text)
    namespace: Dict[str, Any] = {}
    exec(compile(text, f"{key}:parameters", "exec"), namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def interpolate(command: str, namespace: Dict[str, Any]) -> str:
    """IPython's rule for `!cmd {name}`; the notebooks only use plain variable names."""
    def sub(m: "re.Match[str]") -> str:
        name = m.group(1).strip()
        if not name.isidentifier():
            raise ValueError(f"not a plain variable name: {{{m.group(1)}}}")
        return str(namespace[name])
    return INTERPOLATION.sub(sub, command)


def argv_of(command: str, namespace: Dict[str, Any]) -> List[str]:
    """The argument list the shell would build (the interpreter name is replaced by the running one)."""
    argv = shlex.split(interpolate(command, namespace))
    if argv and argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    return argv


def code_cell_with(key: str, needle: str) -> str:
    """The one code cell that holds `needle` (a test that names a cell by its content survives a reorder)."""
    found = [c for c in code_cells(key) if needle in c]
    if len(found) != 1:
        raise KeyError(f"{key}: {len(found)} code cells hold {needle!r}, expected 1")
    return found[0]


def python_source(key: str) -> str:
    """All the code cells as one Python module, in order, with each `!` line turned into `pass`."""
    out = []
    for text in code_cells(key):
        for line in text.split("\n"):
            stripped = line.lstrip()
            out.append(line[: len(line) - len(stripped)] + "pass" if stripped.startswith("!") else line)
        out.append("")
    return "\n".join(out)


def pinned_sha(key: str) -> str:
    m = re.search(r'^SIMPLICIO_SHA\s*=\s*"([^"]*)"', code_cells(key)[0], re.M)
    return m.group(1) if m else ""
