import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

FILE = "analytics.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_row_is_appended_to_a_new_frame():
    m = load()
    df = pd.DataFrame([{"a": 1, "b": 2}])
    out = m.add_row(df, {"a": 3, "b": 4})
    assert len(out) == 2
    assert out.iloc[-1].tolist() == [3, 4]
    assert len(df) == 1
