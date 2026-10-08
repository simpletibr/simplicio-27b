import importlib.util
import os
import sys
from pathlib import Path
import asyncio
import inspect

FILE = "crawler.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_fetch_page_is_a_coroutine_function():
    m = load()
    assert inspect.iscoroutinefunction(m.fetch_page)


def test_awaiting_fetch_page_returns_the_page_text():
    m = load()
    assert asyncio.run(m.fetch_page("u")) == "<html>u</html>"
