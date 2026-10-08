import importlib.util
import os
import sys
from pathlib import Path

FILE = "auth_user.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_sharp_s_matches_ss():
    m = load()
    assert m.compare_usernames("straße", "STRASSE") is True


def test_ascii_case_is_ignored():
    m = load()
    assert m.compare_usernames("Ana", "ANA") is True


def test_different_names_do_not_match():
    m = load()
    assert m.compare_usernames("ana", "anb") is False
