# svg parsing helpers
# src/utils/svg.py
from __future__ import annotations
from pathlib import Path
import re

def is_vector_rich(svg_path: str | Path) -> bool:
    """Heuristic: count path/line/poly elements quickly."""
    txt = Path(svg_path).read_text(encoding="utf-8", errors="ignore")
    cnt = len(re.findall(r"<(path|line|polyline|polygon)\\b", txt, flags=re.I))
    return cnt > 500  # tweakable

def count_svg_text(svg_path: str | Path) -> int:
    txt = Path(svg_path).read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"<text\\b", txt, flags=re.I))
