import ast
import importlib.util
import os
import sys
from pathlib import Path

FILE = "clean_remove.py"  # igual ao campo "edit_file" do task.json


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


def test_existing_file_is_removed(tmp_path):
    m = load()
    p = tmp_path / "a.txt"
    p.write_text("x")
    m.try_remove(str(p))
    assert not p.exists()


def test_second_call_does_not_raise(tmp_path):
    m = load()
    p = tmp_path / "a.txt"
    p.write_text("x")
    m.try_remove(str(p))
    m.try_remove(str(p))


def test_try_except_is_replaced_by_suppress():
    nodes = list(ast.walk(fn_ast("try_remove")))
    assert not any(isinstance(n, ast.Try) for n in nodes)

    def is_suppress(call):
        f = call.func
        return (isinstance(f, ast.Name) and f.id == "suppress") or (
            isinstance(f, ast.Attribute) and f.attr == "suppress"
        )

    withs = [n for n in nodes if isinstance(n, ast.With)]
    assert any(isinstance(i.context_expr, ast.Call) and is_suppress(i.context_expr) for w in withs for i in w.items)
