import importlib.util
import os
import sys
from pathlib import Path

FILE = "rounder.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rounds_up():
    m = load()
    assert m.round_val(2.6) == 3


def test_rounds_down():
    m = load()
    assert m.round_val(2.4) == 2


def test_rounds_negative_values():
    m = load()
    assert m.round_val(-1.7) == -2


def test_result_is_an_int():
    m = load()
    assert type(m.round_val(2.6)) is int
