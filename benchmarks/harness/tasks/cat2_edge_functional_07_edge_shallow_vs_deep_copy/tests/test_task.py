import importlib.util
import os
import sys
from pathlib import Path

FILE = "state_cloner.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_nested_mutation_does_not_leak_into_the_original():
    m = load()
    s = {"nested": [1], "d": {"k": 1}}
    c = m.copy_state(s)
    c["nested"].append(2)
    c["d"]["k"] = 2
    assert s == {"nested": [1], "d": {"k": 1}}
