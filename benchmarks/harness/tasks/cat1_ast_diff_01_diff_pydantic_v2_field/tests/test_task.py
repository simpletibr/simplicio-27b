import importlib.util
import os
import sys
from pathlib import Path

import pydantic
import pytest

FILE = "user_dto.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_accepts_boundary_lengths():
    m = load()
    assert m.UserDTO(tax_id="1" * 11, name="a").tax_id == "1" * 11
    assert m.UserDTO(tax_id="1" * 14, name="a").tax_id == "1" * 14


@pytest.mark.parametrize("size", [10, 15])
def test_rejects_out_of_range_lengths(size):
    m = load()
    with pytest.raises(pydantic.ValidationError):
        m.UserDTO(tax_id="1" * size, name="a")
