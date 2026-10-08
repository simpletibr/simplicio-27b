import importlib.util
import os
import sys
from pathlib import Path

FILE = "params.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_singleton_param_is_a_one_element_tuple():
    m = load()
    assert m.singleton_param == ("only_item",)
