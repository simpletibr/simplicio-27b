import importlib.util
import os
import sys
from pathlib import Path
import math

FILE = "service.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_hundred_with_tax():
    m = load()
    assert math.isclose(m.calculate_total(100.0), 110.0)


def test_fifty_with_tax():
    m = load()
    assert math.isclose(m.calculate_total(50.0), 55.0)
