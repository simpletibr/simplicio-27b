import importlib.util
import os
import sys
from pathlib import Path

FILE = "jwt_decoder.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_user_is_anonymous():
    m = load()
    assert m.get_user_role({}) == "anonymous"


def test_missing_role_is_anonymous():
    m = load()
    assert m.get_user_role({"user": {}}) == "anonymous"


def test_present_role_is_returned():
    m = load()
    assert m.get_user_role({"user": {"role": "admin"}}) == "admin"
