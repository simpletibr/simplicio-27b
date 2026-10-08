import importlib.util
import os
import sys
from pathlib import Path

FILE = "queue_manager.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_empty_list_returns_none():
    m = load()
    assert m.peek_first([]) is None


def test_first_item_is_returned():
    m = load()
    assert m.peek_first([42, 1]) == 42
