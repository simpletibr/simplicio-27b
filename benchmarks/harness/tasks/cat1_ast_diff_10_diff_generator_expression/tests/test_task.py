import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

FILE = "aggregator.py"  # igual ao campo "edit_file" do task.json


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
    cart = [SimpleNamespace(price=1.5), SimpleNamespace(price=2.0)]
    assert m.cart_total(cart) == 3.5


def test_no_intermediate_list():
    nodes = list(ast.walk(fn_ast("cart_total")))
    assert not any(isinstance(n, (ast.ListComp, ast.List)) for n in nodes)
