# src/schema/serialization.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from typing import Any, Dict, List, Tuple

def _graph_json_path(cfg, pdf_stem: str, page: int) -> Path:
    return Path(cfg.paths.processed) / "graphs" / pdf_stem / f"page-{page}.json"

def _load_nodes(cfg, pdf_stem: str, page: int) -> List[Dict[str, Any]]:
    p = _graph_json_path(cfg, pdf_stem, page)
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj.get("nodes", [])

def export_components_csv(cfg, pdf_stem: str, page: int = 1) -> str:
    """Write a tidy components CSV to exports/<pdf_stem>/components_page-<page>.csv and return its path."""
    nodes = _load_nodes(cfg, pdf_stem, page)
    rows = []
    for n in nodes:
        if n.get("kind") != "component":
            continue
        bbox = n.get("bbox") or [0,0,0,0]
        w = max(0.0, bbox[2]-bbox[0]); h = max(0.0, bbox[3]-bbox[1])
        labels = n.get("labels_context") or ""
        if isinstance(labels, list):
            labels = " | ".join(map(str, labels))
        rows.append({
            "id": n.get("id"),
            "type": n.get("type"),
            "confidence": n.get("confidence"),
            "net_id": n.get("net_id"),
            "net_phase": n.get("net_phase"),
            "net_voltage": n.get("net_voltage"),
            "bbox_x0": bbox[0], "bbox_y0": bbox[1], "bbox_x1": bbox[2], "bbox_y1": bbox[3],
            "width": w, "height": h, "area": w*h,
            "labels_context": labels,
        })
    df = pd.DataFrame(rows)
    out_dir = Path(cfg.paths.exports) / pdf_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"components_page-{page}.csv"
    df.to_csv(out, index=False)
    return str(out)
