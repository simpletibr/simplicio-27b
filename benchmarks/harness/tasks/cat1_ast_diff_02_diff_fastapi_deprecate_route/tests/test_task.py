import importlib.util
import os
import sys
from pathlib import Path

FILE = "router.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_route_is_marked_deprecated():
    m = load()
    assert m.app.openapi()["paths"]["/v1/users"]["get"].get("deprecated") is True


def test_response_is_unchanged():
    m = load()
    assert m.list_users() == []
