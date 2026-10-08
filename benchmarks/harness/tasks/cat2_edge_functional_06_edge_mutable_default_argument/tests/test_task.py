import importlib.util
import os
import sys
from pathlib import Path

FILE = "accumulator.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_default_container_is_not_shared():
    m = load()
    c1 = m.collect_data(1)
    c2 = m.collect_data(2)
    assert c1 == [1]
    assert c2 == [2]


def test_explicit_container_is_used():
    m = load()
    assert m.collect_data(3, [0]) == [0, 3]
