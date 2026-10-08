import importlib.util
import os
import sys
from pathlib import Path
import logging

FILE = "worker.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_message_is_built_from_logging_arguments(caplog):
    m = load()
    caplog.set_level(logging.INFO)
    m.process(7, 9)
    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.getMessage() == "Processing job 7 for user 9"
    assert rec.args == (7, 9)
