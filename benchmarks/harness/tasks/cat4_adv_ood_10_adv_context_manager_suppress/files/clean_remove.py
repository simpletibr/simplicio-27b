import os
def try_remove(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
