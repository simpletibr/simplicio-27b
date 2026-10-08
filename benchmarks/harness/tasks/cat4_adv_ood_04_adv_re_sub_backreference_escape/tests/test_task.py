import importlib.util
import os
import sys
from pathlib import Path

FILE = "sanitizer.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_single_number_keeps_the_area_code():
    m = load()
    assert m.redact_phones("Call 555-1234") == "Call 555-XXXX"


def test_every_number_is_redacted():
    m = load()
    assert m.redact_phones("a 555-1234 b 777-9999") == "a 555-XXXX b 777-XXXX"
