import importlib.util
import os
import sys
from pathlib import Path
import inspect

FILE = "shadow.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_returns_the_length():
    m = load()
    assert m.process_items([1, 2, 3]) == 3


def test_parameter_is_named_items():
    m = load()
    assert list(inspect.signature(m.process_items).parameters) == ["items"]


def test_no_local_variable_shadows_len():
    m = load()
    assert "len" not in m.process_items.__code__.co_varnames
