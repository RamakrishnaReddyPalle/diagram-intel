# src/summarize/subsystem_summary.py
from __future__ import annotations
from pathlib import Path
import re, pandas as pd
from src.graph.queries import load_graph

LOAD_CLASSES = {
    "LIFT":     [r"\blift\b", r"\belevator\b"],
    "EV":       [r"\bev\s*charger\b"],
    "HVAC/VRV": [r"\bhvac\b", r"\bvrv\b", r"\bair\s*cond\b", r"\bahu\b"],
    "LIGHTING": [r"\blighting\b", r"\blights\b"],
    "POWER":    [r"\bpower\b", r"\bsocket\b"],
    "FLOOR":    [r"\bfirst\s*floor\b|\bsecond\s*floor\b|\bfloor\b"],
    "SPARE":    [r"\bspare\b"],
}

def _match_any(pats, s) -> bool:
    return any(re.search(p, s, re.I) for p in pats)

def write_subsystem_summary(cfg, pdf_stem: str, page: int = 1) -> Path:
    G = load_graph(cfg, pdf_stem, page)
    rows = []
    for n, a in G.nodes(data=True):
        s = a.get("labels_context")
        if isinstance(s, list): s = " | ".join(map(str,s))
        s = (s or "")
        hits = [k for k,pats in LOAD_CLASSES.items() if _match_any(pats, s)]
        if hits:
            rows.append({"node": n, "class": "|".join(hits), "labels": s, "net_id": a.get("net_id")})
    df = pd.DataFrame(rows).sort_values("class")
    out_dir = Path(cfg.paths.exports) / pdf_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"subsystems_page-{page}.csv"
    df.to_csv(p, index=False)
    return p
