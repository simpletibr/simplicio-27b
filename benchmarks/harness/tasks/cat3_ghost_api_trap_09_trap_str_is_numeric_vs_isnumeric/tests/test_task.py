import importlib.util
import os
import sys
from pathlib import Path

FILE = "cleaner.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_digits_only():
    m = load()
    assert m.is_all_digits("123") is True


def test_letters_are_rejected():
    m = load()
    assert m.is_all_digits("12a") is False


def test_empty_string_is_rejected():
    m = load()
    assert m.is_all_digits("") is False
