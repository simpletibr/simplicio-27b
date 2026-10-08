import ast
import importlib.util
import os
import sys
from pathlib import Path

FILE = "item_lookup.py"  # igual ao campo "edit_file" do task.json


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


def test_behavior_is_unchanged():
    m = load()
    assert m.find_by_id(1) is None


def test_return_annotation_uses_the_union_operator():
    returns = fn_ast("find_by_id").returns
    assert isinstance(returns, ast.BinOp)
    assert isinstance(returns.op, ast.BitOr)


def test_optional_import_is_removed():
    tree = ast.parse((Path(os.environ["TASK_SRC"]) / FILE).read_text())
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
    assert "Optional" not in imported
