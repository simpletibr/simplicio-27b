import importlib.util
import os
import sys
from pathlib import Path

import pytest

FILE = "settings.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", ["true", "1", "yes"])
def test_truthy_strings_become_true(monkeypatch, value):
    monkeypatch.setenv("DEBUG", value)
    m = load()
    assert m.DEBUG is True


@pytest.mark.parametrize("value", ["false", "0", ""])
def test_falsy_strings_become_false(monkeypatch, value):
    monkeypatch.setenv("DEBUG", value)
    m = load()
    assert m.DEBUG is False


def test_unset_variable_is_false(monkeypatch):
    monkeypatch.delenv("DEBUG", raising=False)
    m = load()
    assert m.DEBUG is False
