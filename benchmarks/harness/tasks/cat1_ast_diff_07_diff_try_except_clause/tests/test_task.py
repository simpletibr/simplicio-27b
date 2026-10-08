import importlib.util
import os
import sys
from pathlib import Path

FILE = "storage.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_file_returns_empty_string(tmp_path):
    m = load()
    assert m.read_file(str(tmp_path / "missing.txt")) == ""


def test_existing_file_returns_its_content(tmp_path):
    m = load()
    p = tmp_path / "a.txt"
    p.write_text("abc")
    assert m.read_file(str(p)) == "abc"
