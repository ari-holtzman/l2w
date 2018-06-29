import sys
from contextlib import contextmanager

@contextmanager
def doing(action):
    print("%s..." % action, end='')
    sys.stdout.flush()
    yield
    print("done")
