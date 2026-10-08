import importlib.util
import os
import sys
from pathlib import Path
import warnings

FILE = "async_boot.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def answer():
    return 42


def test_runs_the_coroutine_to_completion_twice():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = load()
        assert m.run_task(answer()) == 42
        assert m.run_task(answer()) == 42
