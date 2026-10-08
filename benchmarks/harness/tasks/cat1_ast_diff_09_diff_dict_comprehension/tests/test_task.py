import ast
import importlib.util
import os
import sys
from pathlib import Path

FILE = "mapper.py"  # igual ao campo "edit_file" do task.json


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


def test_result_is_unchanged():
    m = load()
    assert m.build_lookup([("a", "x"), ("b", "y")]) == {"a": "X", "b": "Y"}


def test_body_is_a_single_dict_comprehension():
    nodes = list(ast.walk(fn_ast("build_lookup")))
    assert not any(isinstance(n, ast.For) for n in nodes)
    assert sum(isinstance(n, ast.DictComp) for n in nodes) == 1
