# table/graph visualizers
# src/ui/components.py
from __future__ import annotations
from pathlib import Path
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd

def _norm_text(x) -> str:
    if x is None: return ""
    if isinstance(x, (list, tuple)): x = " | ".join(map(str, x))
    return str(x)

def _infer_type_from_labels(labels_context: str, typing_hints: Dict[str, List[str]]|None=None) -> str|None:
    if not labels_context: return None
    txt = labels_context.lower()
    hints = typing_hints or {}
    # broader, order matters (specific → generic)
    table = [
        ("ACCL", hints.get("ACCL", []) + ["accl", "changeover", "auto changeover", "2pole, 3way", "ats", "selector"]),
        ("RCCB", hints.get("RCCB", []) + ["rccb", "rcd", "elcb"]),
        ("MCCB", hints.get("MCCB", []) + ["mccb"]),
        ("MCB",  hints.get("MCB", [])  + ["mcb"]),
        ("TPN",  hints.get("TPN", [])  + ["tpn", "tp&n", "t p n"]),
        ("SPD",  hints.get("SPD", [])  + ["spd", "surge"]),
        ("CONTACTOR", hints.get("CONTACTOR", []) + ["contactor"]),
        ("ISOLATOR", hints.get("ISOLATOR", []) + ["isolator"]),
        ("TB",   hints.get("TB", [])   + ["tb", "terminal block"]),
    ]
    for typ, keys in table:
        for k in keys:
            if k in txt:
                return typ
    return None

def engineer_component_summary(nodes: List[Tuple[str, Dict[str, Any]]],
                               typing_hints: Dict[str, List[str]]|None=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (components_table, counts_by_type)
    components_table columns: [component_id, type, confidence, bbox, labels_context]
    counts_by_type columns: [type, count]
    """
    comps = []
    for n, a in nodes:
        if a.get("kind") != "component": continue
        t = a.get("type")
        if not t:
            t = _infer_type_from_labels(a.get("labels_context"), typing_hints)
        comps.append({
            "component_id": a.get("comp_id") or n,
            "type": t or "UNKNOWN",
            "confidence": a.get("confidence", 0.0),
            "bbox": a.get("bbox"),
            "labels_context": _norm_text(a.get("labels_context")),
        })
    df = pd.DataFrame(comps).sort_values(["type", "component_id"]).reset_index(drop=True)
    counts = df.groupby("type", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    return df, counts

def engineer_feeder_load_summary(nodes: List[Tuple[str, Dict[str, Any]]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Heuristic: parse labels context for source/feeder and load keywords.
    Returns (sources_table, loads_table)
    """
    SRC = {
        "GRID/EB":  [r"\bgrid\b", r"\beb\b", r"\bfrom\s*meter\b", r"\bmains\b", r"\bincomer\b"],
        "DG":       [r"\bdg\b", r"\bdiesel\s*gen", r"\bgenerator\b"],
        "PV":       [r"\bpv\b", r"\bsolar\b", r"\binverter\b"],
        "UPS":      [r"\bups\b"],
        "STABILIZER":[r"\bstabiliz", r"\bservo\b"],
    }
    LOAD = {
        "LIFT":     [r"\blift\b"],
        "HVAC/VRV": [r"\bhvac\b", r"\bvrv\b", r"\bach\b", r"\bair\s*cond"],
        "LIGHTING": [r"\blight(ing)?\b"],
        "POWER":    [r"\bpower\b", r"\bsocket\b", r"\braw\s*power\b"],
        "EV":       [r"\bev\s*charger\b", r"\bev\b"],
        "HEAT PUMP":[r"\bheat\s*pump\b"],
        "SPARE":    [r"\bspare\b"],
        "FLOOR":    [r"\bfirst\s*floor\b|\bsecond\s*floor\b|\bfloor\b"],
    }
    def any_match(pats, s) -> bool:
        return any(re.search(p, s, flags=re.I) for p in pats)

    src_rows, load_rows = [], []
    for n, a in nodes:
        s = _norm_text(a.get("labels_context"))
        if not s: continue
        src_hit = [name for name,pats in SRC.items() if any_match(pats, s)]
        load_hit= [name for name,pats in LOAD.items() if any_match(pats, s)]
        if src_hit:
            src_rows.append({"node": n, "matched": "|".join(src_hit), "labels": s})
        if load_hit:
            load_rows.append({"node": n, "matched": "|".join(load_hit), "labels": s})

    return pd.DataFrame(src_rows), pd.DataFrame(load_rows)

def nets_overview(nets_payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for n in nets_payload.get("nets", []):
        rows.append({
            "net_id": n.get("net_id"),
            "nodes": n.get("nodes"),
            "components": n.get("components"),
            "ports": n.get("ports"),
            "junctions": n.get("junctions"),
            "phase": n.get("phase"),
            "voltage": n.get("voltage"),
        })
    df = pd.DataFrame(rows).sort_values("nodes", ascending=False)
    return df
