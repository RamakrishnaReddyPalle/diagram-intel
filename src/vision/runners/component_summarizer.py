# src/vision/runners/component_summarizer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

from src.utils.io import ensure_dir, read_json, write_json
from src.graph.queries import load_graph
from src.vision.clients.qwen2vl import qwen_summarize_component

def _make_prompt_input(node_attrs: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "labels_context": node_attrs.get("labels_context",""),
        "type": node_attrs.get("type"),
        "bbox": node_attrs.get("bbox"),
        "net_phase": node_attrs.get("net_phase"),
        "net_voltage": node_attrs.get("net_voltage"),
    }

def summarize_components(cfg, pdf_stem: str, page: int = 1) -> Path:
    G = load_graph(cfg, pdf_stem, page)
    comps = [(n,a) for n,a in G.nodes(data=True) if a.get("kind")=="component"]
    out_rows=[]
    for n,a in comps:
        inp = _make_prompt_input(a)
        summary = qwen_summarize_component(cfg, inp)
        out_rows.append({"node": n, "summary": summary, "meta": inp})
    out_dir = Path(cfg.paths.processed) / "components" / "summaries" / pdf_stem
    ensure_dir(out_dir)
    out_p = out_dir / f"page-{page}.json"
    write_json(out_rows, out_p)
    return out_p
