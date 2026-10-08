import importlib.util
import os
import sys
from pathlib import Path

FILE = "currency.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_tenth_plus_two_tenths_is_three_tenths():
    m = load()
    assert m.add_cents(0.1, 0.2) == 0.3


def test_seven_tenths_plus_one_tenth_is_eight_tenths():
    m = load()
    assert m.add_cents(0.7, 0.1) == 0.8


def test_result_is_a_float():
    m = load()
    assert type(m.add_cents(0.1, 0.2)) is float
