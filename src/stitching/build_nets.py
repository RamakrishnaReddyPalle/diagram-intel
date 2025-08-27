from pathlib import Path
from typing import Dict, Any, List, Tuple
import re
import networkx as nx

from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging
from src.graph.build_graph import build_graph_for_page, assign_net_ids

# --- phase tokens ---
PHASE_TOKENS = {
    "l1":"L1","l2":"L2","l3":"L3","n":"N",
    "r":"R","y":"Y","b":"B","ryb":"RYB",
    "tpn":"TPN","3φ":"3PH","3ph":"3PH","3-phase":"3PH",
    "1φ":"1PH","1ph":"1PH"
}
PHASE_WORDS = {"neutral":"N"}
VOLT_PAT = re.compile(r"(\d{2,4})\s*V", re.I)

def _nearby_text(cfg, page_text: List[Dict[str,Any]], xy: Tuple[float,float], radius: float) -> List[str]:
    x0, y0 = xy
    out = []
    for t in page_text:
        x1, y1, x2, y2 = t["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if max(abs(cx - x0), abs(cy - y0)) <= radius:
            out.append(t["text"])
    return out

def _tokens_from_text(lines: List[str]) -> Dict[str,int]:
    votes = {}
    for s in lines:
        s_low = s.lower()
        for w, tag in PHASE_WORDS.items():
            if w in s_low:
                votes[tag] = votes.get(tag, 0) + 1
        for tok in re.split(r"[^a-z0-9+]+", s_low):
            if not tok: continue
            if tok in PHASE_TOKENS:
                tag = PHASE_TOKENS[tok]
                votes[tag] = votes.get(tag, 0) + 1
        if "l1" in s_low and "l2" in s_low and "l3" in s_low:
            for t in ("L1","L2","L3"):
                votes[t] = votes.get(t, 0) + 1
    return votes

def _infer_phase_labels(cfg, pdf_stem: str, page: int, G: nx.Graph) -> Dict[int, Dict[str,Any]]:
    vec_page = Path(cfg.paths.processed) / "vector_text" / pdf_stem / f"page-{page}.json"
    texts = read_json(vec_page) if vec_page.exists() else []

    radius = float(cfg.graph.phase_label.search_radius_px)
    votes_by_net: Dict[int, Dict[str,int]] = {}
    volts_by_net: Dict[int, List[int]] = {}

    for n, a in G.nodes(data=True):
        if a.get("kind") not in {"port", "junction"}: 
            continue
        xy = a.get("xy")
        if not xy:
            continue
        nid = a.get("net_id")
        if nid is None:
            continue
        lines = _nearby_text(cfg, texts, tuple(xy), radius)
        v = _tokens_from_text(lines)
        if v:
            d = votes_by_net.setdefault(nid, {})
            for k, c in v.items():
                d[k] = d.get(k, 0) + c
        for s in lines:
            m = VOLT_PAT.search(s)
            if m:
                volts_by_net.setdefault(nid, []).append(int(m.group(1)))

    info: Dict[int, Dict[str,Any]] = {}
    for nid in set(a.get("net_id") for _, a in G.nodes(data=True) if "net_id" in a):
        votes = votes_by_net.get(nid, {})
        tags = [k for k, v in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
                if v >= int(cfg.graph.phase_label.min_token_votes)]
        tagset = set(tags)
        phase = None
        if {"L1","L2","L3"} & tagset:
            phase = "/".join([t for t in ["L1","L2","L3","N"] if t in tagset])
        elif {"R","Y","B"} & tagset:
            phase = "/".join([t for t in ["R","Y","B","N"] if t in tagset])
        elif "TPN" in tagset or "3PH" in tagset:
            phase = "3PH"
        elif "1PH" in tagset:
            phase = "1PH"

        volt = None
        vv = volts_by_net.get(nid, [])
        if vv:
            volt = max(set(vv), key=vv.count)

        info[nid] = {"phase": phase, "voltage": volt, "votes": votes}
        # project back onto node attrs for convenience
        for n, a in G.nodes(data=True):
            if a.get("net_id") == nid:
                a["net_phase"] = phase
                a["net_voltage"] = volt
    return info

def _net_summary(G: nx.Graph) -> Dict[int, Dict[str,Any]]:
    out: Dict[int, Dict[str,Any]] = {}
    for n, a in G.nodes(data=True):
        nid = a.get("net_id")
        if nid is None:
            continue
        d = out.setdefault(nid, {
            "nodes": 0, "components": 0, "ports": 0, "junctions": 0,
            "phase": a.get("net_phase"), "voltage": a.get("net_voltage")
        })
        d["nodes"] += 1
        k = a.get("kind")
        if k == "component":
            d["components"] += 1
        elif k == "port":
            d["ports"] += 1
        elif k == "junction":
            d["junctions"] += 1
    return out


def stitch_page(cfg, pdf_stem: str, page: int = 1):
    log = setup_logging(cfg.logging.level)
    from src.graph.exporters import export_graph

    G = build_graph_for_page(cfg, pdf_stem, page=page)
    assign_net_ids(G)
    net_info = _infer_phase_labels(cfg, pdf_stem, page, G)

    # write nets summary
    nets = _net_summary(G)
    out_dir = Path(cfg.paths.processed) / "nets" / pdf_stem
    ensure_dir(out_dir)
    payload = {
        "pdf": pdf_stem,
        "page": page,
        "count": len(nets),
        "nets": [{"net_id": nid, **d} for nid, d in sorted(nets.items(), key=lambda kv: kv[0])]
    }
    write_json(payload, out_dir / f"page-{page}.json")
    log.info(f"[stitch] {pdf_stem} page-{page}: nets={len(nets)} → {out_dir/f'page-{page}.json'}")

    # export graph
    export_graph(cfg, pdf_stem, page, G)
    return G, payload
