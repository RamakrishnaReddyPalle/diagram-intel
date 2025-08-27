# src/ingest/rasterize.py
from __future__ import annotations
from pathlib import Path
import subprocess
from src.utils.io import ensure_dir

def pdftocairo_png(pdf_path: str, out_png: str, dpi: int = 600) -> bool:
    out_png = str(out_png)
    ensure_dir(Path(out_png).parent)
    cmd = ["pdftocairo","-png","-singlefile","-r",str(dpi),pdf_path,out_png[:-4]]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False
