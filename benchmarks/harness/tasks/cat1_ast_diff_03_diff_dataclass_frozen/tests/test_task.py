import importlib.util
import os
import sys
from pathlib import Path

import pytest

FILE = "config.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_default_port():
    m = load()
    assert m.AppConfig().port == 8080


def test_assignment_after_construction_raises():
    m = load()
    cfg = m.AppConfig()
    with pytest.raises(AttributeError):
        cfg.port = 1
