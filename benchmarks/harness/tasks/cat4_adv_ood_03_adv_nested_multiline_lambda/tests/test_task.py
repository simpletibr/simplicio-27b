import importlib.util
import os
import sys
from pathlib import Path

FILE = "sort.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_sorts_in_place_and_returns_the_same_list():
    m = load()
    xs = [{"id": 2, "v": "b"}, {"id": 1, "v": "a"}]
    assert m.sort_by_id(xs) is xs
    assert [d["id"] for d in xs] == [1, 2]
