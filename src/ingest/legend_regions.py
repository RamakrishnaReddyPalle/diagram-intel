# src/ingest/legend_regions.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
from src.utils.io import read_json

KEYWORDS = ("Title:", "Drawing No", "Rev", "Prepared", "Checked", "Approved")

def detect_legend_bbox(cfg, pdf_stem: str, page: int = 1) -> Optional[Tuple[float,float,float,float]]:
    """Heuristic: find cluster of metadata labels; return bbox union."""
    vec = Path(cfg.paths.processed)/"vector_text"/pdf_stem/f"page-{page}.json"
    if not vec.exists(): return None
    items = read_json(vec)
    hits = [it for it in items if any(k.lower() in it["text"].lower() for k in KEYWORDS)]
    if not hits: return None
    x1=min(it["bbox"][0] for it in hits); y1=min(it["bbox"][1] for it in hits)
    x2=max(it["bbox"][2] for it in hits); y2=max(it["bbox"][3] for it in hits)
    # inflate a bit
    return (x1-50, y1-50, x2+50, y2+50)
