import importlib.util
import os
import sys
from pathlib import Path

FILE = "slug_validator.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_trailing_newline_is_rejected():
    m = load()
    assert m.is_valid_slug("valid-slug\n") is False


def test_valid_slug_is_accepted():
    m = load()
    assert m.is_valid_slug("valid-slug") is True


def test_invalid_characters_are_rejected():
    m = load()
    assert m.is_valid_slug("Bad Slug") is False
