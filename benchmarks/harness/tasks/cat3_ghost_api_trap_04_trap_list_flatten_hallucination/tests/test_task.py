import importlib.util
import os
import sys
from pathlib import Path

FILE = "arrays.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_flattens_in_order():
    m = load()
    assert m.flatten([[1, 2], [3], []]) == [1, 2, 3]


def test_empty_input():
    m = load()
    assert m.flatten([]) == []
