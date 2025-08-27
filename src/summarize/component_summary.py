# src/summarize/component_summary.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re
import pandas as pd

from src.utils.io import read_json

# --- heuristics --------------------------------------------------------------

# Regex shortcuts for common devices (kept intentionally small & transparent)
_PATTERNS = {
    "MCCB":       re.compile(r"\bMCCB\b", re.I),
    "RCCB":       re.compile(r"\b(RCCB|RCD|ELCB)\b", re.I),
    "MCB":        re.compile(r"\bMCB\b", re.I),
    "TPN":        re.compile(r"\bTPN\b", re.I),
    "Isolator":   re.compile(r"\bIsolator\b", re.I),
    "SPD":        re.compile(r"\bSPD\b", re.I),
    "Terminal Block": re.compile(r"\bTB(\d+|-\d+)?\b|\bTerminal\s*Block\b", re.I),
    "ACCL":       re.compile(r"\bACCL\b|Automatic\s*Change\s*Over", re.I),
    "Contactor":  re.compile(r"\bContactor\b", re.I),
    "Meter":      re.compile(r"\bMeter\b", re.I),
    "Inverter":   re.compile(r"\bInverter\b|\bPV\b", re.I),
    "Stabilizer": re.compile(r"\bStabilizer\b", re.I),
    # common loads (nice to list; you may drop these if you only want switchgear)
    "Lift":       re.compile(r"\bLift\b", re.I),
    "VRV":        re.compile(r"\bVRV\b", re.I),
    "EV Charger": re.compile(r"\bEV\s*Charger\b", re.I),
    "Heat Pump":  re.compile(r"\bHeat\s*Pump\b", re.I),
    "Solar":      re.compile(r"\bSolar\b", re.I),
}

# Ratings like "100A", "230V", "25kA", "30mA" etc.
_RATING_RE = re.compile(r"\b(\d+)\s*(A|V|kA|kV|mA)\b", re.I)

def _textify(v) -> str:
    if isinstance(v, (list, tuple)):
        return " | ".join(str(x) for x in v if x is not None)
    return str(v or "")

def _infer_device(text: str, type_hint: str | None, hints_dict: Dict[str, List[str]] | None) -> str:
    t = text.lower()
    # 1) typing_hints_contains from constraints packs
    if hints_dict:
        for name, tokens in hints_dict.items():
            if any(tok.lower() in t for tok in (tokens or [])):
                return name.upper()
    # 2) simple regex patterns
    for label, pat in _PATTERNS.items():
        if pat.search(text):
            return label
    # 3) fallback to model-assigned type, else Un-typed
    return (type_hint or "Un-typed")

def _extract_ratings(text: str) -> str:
    # return a short, de-duplicated CSV like "100A, 30mA"
    vals = []
    seen = set()
    for m in _RATING_RE.finditer(text):
        s = (m.group(1) + m.group(2)).upper().replace("KA", "kA").replace("KV","kV")
        if s not in seen:
            seen.add(s)
            vals.append(s)
    return ", ".join(vals[:4])  # keep it short

# --- public API --------------------------------------------------------------

def build_device_inventory(cfg, pdf_stem: str, page: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      - inventory_df: one row per device type with Qty, Typical rating, Example labels
      - details_df:  one row per component (Device, Phase, Ratings, Labels, bbox_w, bbox_h)
    It prefers merged components; if unavailable, falls back to graph nodes.
    """
    hints = (getattr(cfg.constraints, "inference", {}) or {}).get("typing_hints_contains", {})
    merged_idx_path = Path(cfg.paths.processed) / "components" / "merged" / "merged.index.json"
    rows: List[Dict[str, Any]] = []

    if merged_idx_path.exists():
        merged_idx = read_json(merged_idx_path)
        for rec in merged_idx:
            if rec.get("pdf") != pdf_stem or int(rec.get("page", 1)) != page:
                continue
            comp = read_json(rec["path"])
            labels = _textify(comp.get("labels_context"))
            dev = _infer_device(labels, comp.get("type"), hints)
            ratings = _extract_ratings(labels)
            phase = comp.get("net_phase") or "-"
            bbox = comp.get("bbox") or comp.get("tile_bbox") or [0,0,0,0]
            w = max(0.0, float(bbox[2]-bbox[0])) if len(bbox)>=4 else 0.0
            h = max(0.0, float(bbox[3]-bbox[1])) if len(bbox)>=4 else 0.0
            rows.append({
                "Device": dev,
                "Phase": phase,
                "Ratings": ratings or "-",
                "Labels": labels[:180] + ("…" if len(labels) > 180 else ""),  # keep short
                "bbox_w": round(w),
                "bbox_h": round(h),
            })
    else:
        # fallback: graph nodes
        gpath = Path(cfg.paths.processed) / "graphs" / pdf_stem / f"page-{page}.json"
        G = read_json(gpath)
        for node in G.get("nodes", []):
            if node.get("kind") != "component":
                continue
            labels = _textify(node.get("labels_context"))
            dev = _infer_device(labels, node.get("type"), hints)
            ratings = _extract_ratings(labels)
            phase = node.get("net_phase") or "-"
            bbox = node.get("bbox") or [0,0,0,0]
            w = max(0.0, float(bbox[2]-bbox[0])) if len(bbox)>=4 else 0.0
            h = max(0.0, float(bbox[3]-bbox[1])) if len(bbox)>=4 else 0.0
            rows.append({
                "Device": dev,
                "Phase": phase,
                "Ratings": ratings or "-",
                "Labels": labels[:180] + ("…" if len(labels) > 180 else ""),
                "bbox_w": round(w),
                "bbox_h": round(h),
            })

    if not rows:
        return (pd.DataFrame(columns=["Device","Qty","Typical rating","Example labels"]),
                pd.DataFrame(columns=["Device","Phase","Ratings","Labels","bbox_w","bbox_h"]))

    details_df = pd.DataFrame(rows)
    # Build inventory aggregation
    # Choose the most common rating string within each device as "Typical rating"
    agg = (details_df
           .groupby("Device", dropna=False)
           .agg(Qty=("Device","size"),
                _ratings=("Ratings", lambda s: s.value_counts().index[0] if len(s)>0 else "-"),
                _sample=("Labels",  lambda s: s.iloc[0] if len(s)>0 else "-"))
           .reset_index()
           .rename(columns={"_ratings":"Typical rating","_sample":"Example labels"})
           .sort_values(["Qty","Device"], ascending=[False, True])
           )
    return agg, details_df

# Backwards-compat thin wrappers (used by your app before)
def component_counts(cfg, pdf_stem: str, page: int = 1) -> Dict[str,int]:
    inv, _ = build_device_inventory(cfg, pdf_stem, page)
    return {row["Device"]: int(row["Qty"]) for _, row in inv.iterrows()}

def summarize_components(cfg, pdf_stem: str, page: int = 1, top_n: int = 6) -> List[str]:
    # Produce human-friendly one-liners from details (largest by bbox area as proxy)
    _, details = build_device_inventory(cfg, pdf_stem, page)
    if details.empty:
        return []
    details = details.assign(area=details["bbox_w"].fillna(0)*details["bbox_h"].fillna(0))
    top = details.sort_values("area", ascending=False).head(top_n)
    out = []
    for _, r in top.iterrows():
        out.append(f"{r['Device']} — phase={r['Phase']} rating={r['Ratings']} | labels: {r['Labels']}")
    return out
