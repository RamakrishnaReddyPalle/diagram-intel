from pathlib import Path
from typing import Dict, Any, List, Tuple
import networkx as nx

from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging

def _project_to_edge(px, py, bb, prefer_side=None):
    x1,y1,x2,y2 = bb
    # pick the closest edge by L-inf unless prefer_side given
    candidates = [
        ("left",   (x1, py)),
        ("right",  (x2, py)),
        ("top",    (px, y1)),
        ("bottom", (px, y2)),
    ]
    if prefer_side:
        # ensure preferred edge is first
        candidates.sort(key=lambda c: 0 if c[0]==prefer_side else 1)
    # choose min distance
    best = None
    bestd = 1e9
    for side,(qx,qy) in candidates:
        d = max(abs(px-qx), abs(py-qy))
        if d < bestd:
            best = (side,(qx,qy)); bestd = d
    return best  # (side,(qx,qy))

def autofix_ports_on_edges(cfg, pdf_stem: str, page: int = 1):
    log = setup_logging(cfg.logging.level)
    graph_json = Path(cfg.paths.processed)/"graphs"/pdf_stem/f"page-{page}.json"
    vio_json   = Path(cfg.paths.processed)/"refine"/pdf_stem/f"page-{page}.violations.json"
    assert graph_json.exists(), f"missing graph: {graph_json}"
    assert vio_json.exists(),   f"missing violations: {vio_json} (run detector first)"

    Gp = read_json(graph_json)
    vios = read_json(vio_json)["violations"]

    # materialize graph
    import networkx as nx
    G = nx.Graph(**Gp.get("graph_attrs", {}))
    for n in Gp["nodes"]:
        nid = n.pop("id"); G.add_node(nid, **n)
    for e in Gp["edges"]:
        u = e.pop("u"); v = e.pop("v"); G.add_edge(u, v, **e)

    fixed = 0
    for v in vios:
        if v["type"] != "port_off_edge": 
            continue
        n = v["node"]
        # prefer the side the port was originally tagged with, if any
        side_pref = G.nodes[n].get("side")
        side, (qx,qy) = _project_to_edge(G.nodes[n]["xy"][0], G.nodes[n]["xy"][1], tuple(v["bbox"]), side_pref)
        G.nodes[n]["xy"] = [int(qx), int(qy)]
        G.nodes[n]["side"] = side
        fixed += 1

    # write a refined graph json (v1: JSON only)
    out_dir = Path(cfg.paths.processed)/"graphs_refined"/pdf_stem
    ensure_dir(out_dir)
    payload = {
        "graph_attrs": G.graph,
        "nodes": [{"id": n, **attrs} for n, attrs in G.nodes(data=True)],
        "edges": [{"u": u, "v": v, **attrs} for u, v, attrs in G.edges(data=True)],
    }
    out_path = out_dir/f"page-{page}.json"
    write_json(payload, out_path)
    log.info(f"[refine.autofix] moved {fixed} ports to bbox edges → {out_path}")
    return {"fixed": fixed, "path": str(out_path)}
