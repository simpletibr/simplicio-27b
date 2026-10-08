import importlib.util
import os
import sys
from pathlib import Path
import threading

FILE = "counter.py"  # igual ao campo "edit_file" do task.json


def load():
    path = Path(os.environ["TASK_SRC"]) / FILE
    name = "task_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SpyLock:
    def __init__(self):
        self.n = 0

    def acquire(self, *a, **k):
        self.n += 1
        return True

    def release(self):
        pass

    def __enter__(self):
        self.n += 1
        return self

    def __exit__(self, *a):
        return False


def test_inc_uses_lock():
    m = load()
    c = m.Counter()
    lock_types = (type(threading.Lock()), type(threading.RLock()))
    names = [k for k, v in vars(c).items() if isinstance(v, lock_types)]
    assert len(names) == 1
    spy = SpyLock()
    setattr(c, names[0], spy)
    c.inc()
    assert spy.n == 1 and c.val == 1
