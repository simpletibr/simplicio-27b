"""Checks the eval tasks against the training data: no task may share a 13-gram with it.

For every task directory under benchmarks/harness/tasks/ that has a task.json, the text
`instruction + "\\n" + files/<edit_file> + "\\n" + oracle.txt` is split into tokens and its
13-grams are intersected with the 13-grams of every string, scanned recursively, of every
JSON line of data/*.jsonl. The result is written to benchmarks/harness/tasks/decontamination.json.
Standard library only. Exit code 1 when any task overlaps.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

N = 13
TOKENIZER = r"re.findall(r'\w+', text.lower())"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = HERE / "tasks"
DATA = ROOT / "data"
REPORT = "decontamination.json"
MAX_EXAMPLES = 3


def tokens(text):
    return re.findall(r"\w+", text.lower())


def ngrams(text, n=N):
    toks = tokens(text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def corpus_files(data_dir):
    return sorted(Path(data_dir).glob("*.jsonl"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_text(task_dir):
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    code = (task_dir / "files" / task["edit_file"]).read_text(encoding="utf-8")
    oracle = (task_dir / "oracle.txt").read_text(encoding="utf-8")
    return task["instruction"] + "\n" + code + "\n" + oracle


def task_dirs(tasks_dir):
    return sorted(d for d in Path(tasks_dir).iterdir() if d.is_dir() and (d / "task.json").is_file())


def build_report(tasks_dir=TASKS, data_dir=DATA, root=ROOT):
    grams = {}
    index = {}
    for d in task_dirs(tasks_dir):
        grams[d.name] = ngrams(task_text(d))
        for gram in grams[d.name]:
            index.setdefault(gram, []).append(d.name)
    found = {name: set() for name in grams}
    files = corpus_files(data_dir)
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                for text in strings(json.loads(line)):
                    for gram in ngrams(text) & index.keys():
                        for name in index[gram]:
                            found[name].add(gram)
    return {
        "n": N,
        "tokenizer": TOKENIZER,
        "corpus": [{"path": Path(p).resolve().relative_to(Path(root).resolve()).as_posix(), "sha256": sha256(p)} for p in files],
        "tasks": [
            {
                "id": name,
                "overlaps": len(found[name]),
                "examples": [" ".join(g) for g in sorted(found[name])[:MAX_EXAMPLES]],
            }
            for name in grams
        ],
    }


def main():
    report = build_report()
    (TASKS / REPORT).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    bad = [t["id"] for t in report["tasks"] if t["overlaps"]]
    print(f"decontamination: {len(report['tasks'])} tasks, {len(bad)} with overlap {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
