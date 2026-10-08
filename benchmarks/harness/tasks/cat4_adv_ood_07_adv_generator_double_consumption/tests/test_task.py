import importlib.util
import os
import sys
from pathlib import Path

FILE = "stream.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_generator_input():
    m = load()

    def gen():
        yield 10
        yield 20

    assert m.analyze_stream(gen()) == (2, 30)


def test_list_input():
    m = load()
    assert m.analyze_stream([1, 2, 3]) == (3, 6)
