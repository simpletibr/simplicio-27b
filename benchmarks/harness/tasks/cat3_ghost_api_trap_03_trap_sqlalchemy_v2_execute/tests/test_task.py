import importlib.util
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

FILE = "db.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StrictSession(Session):
    def query(self, *args, **kwargs):
        raise AssertionError("legacy Session.query() must not be used")


def test_get_users_returns_every_row_as_a_list():
    m = load()
    engine = create_engine("sqlite://")
    m.Base.metadata.create_all(engine)
    with StrictSession(engine, expire_on_commit=False) as s:
        s.add(m.User(name="a"))
        s.add(m.User(name="b"))
        s.commit()
        users = m.get_users(s)
        assert isinstance(users, list)
        assert sorted(u.name for u in users) == ["a", "b"]
