from pathlib import Path
from typing import Dict, Any, Tuple, List
import re
import networkx as nx

from src.utils.io import read_json
from src.utils.logging import setup_logging

# simple token maps
PHASE_TOKENS = {
    "l1":"L1","l2":"L2","l3":"L3","n":"N",
    "r":"R","y":"Y","b":"B","ryb":"RYB","tpn":"TPN","3φ":"3PH","3ph":"3PH","3-phase":"3PH","1φ":"1PH","1ph":"1PH"
}
PHASE_WORDS = {
    "neutral":"N"
}
VOLT_PAT = re.compile(r"(\d{2,4})\s*V", re.I)

def _nearby_text(cfg, page_text: List[Dict[str,Any]], xy: Tuple[float,float], radius: float) -> List[str]:
    x0,y0 = xy
    out=[]
    for t in page_text:
        x1,y1,x2,y2 = t["bbox"]
        # center of text bbox
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        if max(abs(cx-x0), abs(cy-y0)) <= radius:
            out.append(t["text"])
    return out

def _tokens_from_text(lines: List[str]) -> Dict[str,int]:
    votes={}
    for s in lines:
        s_low = s.lower()
        # words
        for w,tag in PHASE_WORDS.items():
            if w in s_low:
                votes[tag] = votes.get(tag,0)+1
        # single/double tokens
        # split on non-alnum
        for tok in re.split(r"[^a-z0-9+]+", s_low):
            if not tok: continue
            if tok in PHASE_TOKENS:
                tag = PHASE_TOKENS[tok]
                votes[tag] = votes.get(tag,0)+1
        # explicit patterns like L1/L2/L3
        if "l1" in s_low and "l2" in s_low and "l3" in s_low:
            votes["L1"]=votes.get("L1",0)+1
            votes["L2"]=votes.get("L2",0)+1
            votes["L3"]=votes.get("L3",0)+1
        # voltage
    return votes

def infer_phase_labels(cfg, pdf_stem: str, page: int, G: nx.Graph) -> Dict[int,Dict[str,Any]]:
    log = setup_logging(cfg.logging.level)
    # page vector text
    vec_page = Path(cfg.paths.processed)/"vector_text"/pdf_stem/f"page-{page}.json"
    texts = read_json(vec_page) if vec_page.exists() else []

    # gather votes per net
    radius = float(cfg.graph.phase_label.search_radius_px)
    net_votes: Dict[int, Dict[str,int]] = {}
    volt_votes: Dict[int, List[int]] = {}

    # look around ports & junctions (most reliable)
    for n, attrs in G.nodes(data=True):
        if attrs.get("kind") not in {"port","junction"}: 
            continue
        xy = tuple(attrs.get("xy",(None,None)))
        if xy[0] is None: 
            continue
        n_id = attrs.get("net_id")
        if n_id is None:
            continue
        lines = _nearby_text(cfg, texts, xy, radius)
        votes = _tokens_from_text(lines)
        if votes:
            d = net_votes.setdefault(n_id, {})
            for k,v in votes.items():
                d[k] = d.get(k,0)+v
        # voltage
        for s in lines:
            m = VOLT_PAT.search(s)
            if m:
                volt_votes.setdefault(n_id, []).append(int(m.group(1)))

    # finalize per net
    net_info: Dict[int, Dict[str,Any]] = {}
    for n_id in set(a.get("net_id") for _,a in G.nodes(data=True) if "net_id" in a):
        votes = net_votes.get(n_id, {})
        tags = [k for k,v in sorted(votes.items(), key=lambda kv:(-kv[1], kv[0])) if v >= int(cfg.graph.phase_label.min_token_votes)]
        # normalize: prefer L1/L2/L3/N; else R/Y/B/N
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
        if n_id in volt_votes and volt_votes[n_id]:
            # choose the mode / most common voltage
            vv = volt_votes[n_id]
            volt = max(set(vv), key=vv.count)

        net_info[n_id] = {"phase": phase, "voltage": volt, "votes": votes}

    # write back to graph attributes
    for n,attrs in G.nodes(data=True):
        nid = attrs.get("net_id")
        if nid in net_info:
            attrs["net_phase"] = net_info[nid].get("phase")
            attrs["net_voltage"] = net_info[nid].get("voltage")

    return net_info
