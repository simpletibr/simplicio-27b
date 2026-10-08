import importlib.util
import os
import sys
from pathlib import Path
import builtins

FILE = "file_logger.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_closes(tmp_path, monkeypatch):
    m = load()
    real_open, opened = builtins.open, []

    def spy(*a, **k):
        f = real_open(*a, **k)
        opened.append(f)
        return f

    monkeypatch.setattr(builtins, "open", spy)
    p = tmp_path / "log.txt"
    m.log_message(str(p), "hi")
    m.log_message(str(p), "yo")
    monkeypatch.undo()
    assert opened and all(f.closed for f in opened)
    assert p.read_text() == "hi\nyo\n"
