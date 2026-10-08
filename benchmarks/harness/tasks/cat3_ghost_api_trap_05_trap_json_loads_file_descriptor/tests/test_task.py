import importlib.util
import os
import sys
from pathlib import Path

FILE = "parser.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_returns_the_parsed_json(tmp_path):
    m = load()
    p = tmp_path / "data.json"
    p.write_text('{"a": 1}')
    assert m.parse_file(str(p)) == {"a": 1}
