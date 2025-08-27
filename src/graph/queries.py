# src/graph/queries.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re
import networkx as nx

from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging

def _choose_graph_path(cfg, pdf_stem: str, page: int) -> Path:
    refined = Path(cfg.paths.processed)/"graphs_refined"/pdf_stem/f"page-{page}.json"
    base    = Path(cfg.paths.processed)/"graphs"/pdf_stem/f"page-{page}.json"
    return refined if refined.exists() else base

def load_graph(cfg, pdf_stem: str, page: int = 1) -> nx.Graph:
    payload = read_json(_choose_graph_path(cfg, pdf_stem, page))
    G = nx.Graph()
    for nd in payload.get("nodes", []):
        nid = nd["id"]; attrs = {k:v for k,v in nd.items() if k!="id"}
        G.add_node(nid, **attrs)
    for ed in payload.get("edges", []):
        u, v = ed.get("u"), ed.get("v")
        if u is None or v is None: continue
        attrs = {k:v for k,v in ed.items() if k not in ("u","v")}
        G.add_edge(u, v, **attrs)
    for k,v in payload.get("graph_attrs", {}).items():
        G.graph[k] = v
    return G

def find_nodes_by_text(G: nx.Graph, pattern: str, kinds: Optional[List[str]] = None) -> List[Tuple[str, Dict[str,Any]]]:
    """Regex search over labels_context and type."""
    rx = re.compile(pattern, re.IGNORECASE)
    out = []
    for n,a in G.nodes(data=True):
        if kinds and a.get("kind") not in kinds:
            continue
        blob = ""
        ctx = a.get("labels_context")
        if isinstance(ctx, list): blob += " | ".join([str(x) for x in ctx])
        elif isinstance(ctx, str): blob += ctx
        t = a.get("type")
        if t: blob += f" | {t}"
        if rx.search(blob or ""):
            out.append((n,a))
    return out

def nets_table(cfg, pdf_stem: str, page: int = 1) -> List[Dict[str,Any]]:
    p = Path(cfg.paths.processed)/"nets"/pdf_stem/f"page-{page}.json"
    payload = read_json(p)
    nets = payload.get("nets", [])
    # sorted by size desc
    return sorted(nets, key=lambda d: d.get("nodes", 0), reverse=True)

def shortest_path_between_labels(G: nx.Graph, src_pat: str, dst_pat: str,
                                 kinds: Optional[List[str]] = None) -> Dict[str, Any]:
    srcs = [n for n,_ in find_nodes_by_text(G, src_pat, kinds)]
    dsts = [n for n,_ in find_nodes_by_text(G, dst_pat, kinds)]
    best = {"path": [], "length": None, "src": None, "dst": None}
    for s in srcs:
        for t in dsts:
            try:
                p = nx.shortest_path(G, s, t)
                if best["length"] is None or len(p) < best["length"]:
                    best = {"path": p, "length": len(p), "src": s, "dst": t}
            except nx.NetworkXNoPath:
                continue
    return best

def extract_subgraph(G: nx.Graph, center_node: str, hops: int = 2) -> nx.Graph:
    nodes = set([center_node])
    frontier = {center_node}
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            nxt.update(G.neighbors(u))
        nodes.update(nxt)
        frontier = nxt
    return G.subgraph(nodes).copy()

def save_query_result(cfg, pdf_stem: str, name: str, data: Dict[str,Any], page: int = 1) -> Path:
    out_dir = Path(cfg.paths.processed)/"queries"/pdf_stem
    ensure_dir(out_dir)
    out = out_dir/f"page-{page}.{name}.json"
    write_json(data, out)
    return out

# src/graph/queries.py — add this tiny helper
def summarize_components_simple(cfg, pdf_stem: str, page=1):
    from pathlib import Path
    from src.utils.io import read_json
    idx = read_json(Path(cfg.paths.processed)/"components"/"merged"/"merged.index.json")
    out = []
    for row in idx:
        rec = read_json(Path(row["path"]))
        out.append({"type": rec.get("type"), "labels": rec.get("labels_context","")})
    return out
