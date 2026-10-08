import importlib.util
import os
import sys
from pathlib import Path

FILE = "math_utils.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_zero_divisor_returns_zero():
    m = load()
    assert m.safe_divide(10.0, 0.0) == 0.0


def test_normal_division_is_unchanged():
    m = load()
    assert m.safe_divide(10.0, 2.0) == 5.0
