import importlib.util
import os
import sys
from pathlib import Path

FILE = "string_utils.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_none_returns_empty_string():
    m = load()
    assert m.format_title(None) == ""


def test_blank_returns_empty_string():
    m = load()
    assert m.format_title("   ") == ""


def test_regular_title_is_capitalized():
    m = load()
    assert m.format_title("hello world") == "Hello World"
