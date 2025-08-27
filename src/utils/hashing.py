# src/utils/hashing.py
from pathlib import Path
import hashlib

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def pdf_fingerprint(path: str) -> str:
    p = Path(path)
    return f"{p.stem}-{sha1_file(path)[:8]}"
