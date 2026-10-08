import ast
import importlib.util
import os
import sys
from pathlib import Path

FILE = "finder.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fn_ast(name):
    tree = ast.parse((Path(os.environ["TASK_SRC"]) / FILE).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def test_match_is_returned():
    m = load()
    assert m.find_match(r"\d+", "ab12") == "12"


def test_no_match_returns_none():
    m = load()
    assert m.find_match(r"\d+", "ab") is None


def test_match_is_bound_in_the_if_condition():
    ifs = [n for n in ast.walk(fn_ast("find_match")) if isinstance(n, ast.If)]
    assert any(isinstance(x, ast.NamedExpr) for i in ifs for x in ast.walk(i.test))
