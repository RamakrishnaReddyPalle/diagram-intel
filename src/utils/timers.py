# src/utils/timers.py
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[timer] {name}: {dt:.3f}s")
