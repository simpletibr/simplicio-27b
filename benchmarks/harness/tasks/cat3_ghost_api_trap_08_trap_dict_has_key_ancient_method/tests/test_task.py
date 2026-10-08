import importlib.util
import os
import sys
from pathlib import Path

FILE = "legacy.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_present_key():
    m = load()
    assert m.check_key({"k": 1}, "k") is True


def test_absent_key():
    m = load()
    assert m.check_key({}, "k") is False
