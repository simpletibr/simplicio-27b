import importlib.util
import os
import sys
from pathlib import Path
import warnings

FILE = "schemas.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_name_is_stored_in_uppercase():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = load()
        assert m.User(name="ana").name == "ANA"
