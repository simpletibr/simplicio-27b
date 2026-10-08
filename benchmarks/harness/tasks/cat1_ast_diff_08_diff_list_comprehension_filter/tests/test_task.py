import ast
import importlib.util
import os
import sys
from pathlib import Path

FILE = "filter.py"  # igual ao campo "edit_file" do task.json


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
    assert m.evens_of([1, 2, 3, 4, 6]) == [2, 4, 6]


def test_body_is_a_single_list_comprehension():
    nodes = list(ast.walk(fn_ast("evens_of")))
    assert not any(isinstance(n, ast.For) for n in nodes)
    assert sum(isinstance(n, ast.ListComp) for n in nodes) == 1
