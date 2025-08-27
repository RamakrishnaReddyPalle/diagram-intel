from pathlib import Path
from typing import Dict, Any, List, Tuple
import networkx as nx
from src.utils.io import read_json
from src.utils.logging import setup_logging

Coord = Tuple[int, int]

def _endpoint_to_junction_id(ep: Coord, junctions: List[Dict[str,Any]]) -> str | None:
    # exact membership; fallback to nearest
    for j in junctions:
        if [int(ep[0]), int(ep[1])] in j.get("members", []):
            return j["id"]
    best = (None, 1e9)
    for j in junctions:
        cx, cy = j["xy"]
        d = max(abs(cx - ep[0]), abs(cy - ep[1]))
        if d < best[1]:
            best = (j["id"], d)
    return best[0]

def build_graph_for_page(cfg, pdf_stem: str, page: int = 1) -> nx.Graph:
    log = setup_logging(cfg.logging.level)

    wires_path = Path(cfg.paths.processed) / "wires" / pdf_stem / f"page-{page}.json"
    ports_path = Path(cfg.paths.processed) / "ports" / pdf_stem / f"page-{page}.json"
    merged_idx = Path(cfg.paths.processed) / "components" / "merged" / "merged.index.json"

    assert wires_path.exists(), f"missing wires: {wires_path}"
    assert ports_path.exists(), f"missing ports: {ports_path}"
    assert merged_idx.exists(), "missing merged components index"

    W = read_json(wires_path)
    P = read_json(ports_path)
    merged_list = read_json(merged_idx)
    comps_meta = [read_json(Path(r["path"])) for r in merged_list
                  if r["pdf"] == pdf_stem and int(r["page"]) == page]

    G = nx.Graph(pdf=pdf_stem, page=page)

    # Components
    for c in comps_meta:
        cid = f"comp:{c['id']}"
        G.add_node(cid,
                   kind="component",
                   comp_id=c["id"],
                   bbox=c["bbox"],
                   type=c.get("type"),
                   confidence=float(c.get("confidence") or 0.0),
                   labels_context=" | ".join(c.get("labels_context", [])))

    # Ports (and comp ↔ port edges)
    for p in P["ports"]:
        pid = f"port:{p['comp_id']}:{p['port_id']}"
        G.add_node(pid, kind="port", comp_id=p["comp_id"], port_id=p["port_id"],
                   xy=tuple(p["xy"]), side=p.get("side"))
        G.add_edge(f"comp:{p['comp_id']}", pid, kind="has_port")

    # Junctions
    for j in P["junctions"]:
        jid = f"junc:{j['id']}"
        G.add_node(jid, kind="junction", junc_id=j["id"], xy=tuple(j["xy"]))

    # Port ↔ Junction edges (endpoint snaps)
    for conn in P["connections"]:
        ep = tuple(conn["endpoint"])
        jid_raw = _endpoint_to_junction_id(ep, P["junctions"])
        if not jid_raw:
            continue
        pid = f"port:{conn['comp_id']}:{conn['port_id']}"
        jid = f"junc:{jid_raw}"
        if pid in G and jid in G:
            G.add_edge(pid, jid, kind="wire")

    # Wire segments between junctions
    for poly in W["polylines"]:
        (x1, y1), (x2, y2) = poly["polyline"]
        j1_raw = _endpoint_to_junction_id((x1, y1), P["junctions"])
        j2_raw = _endpoint_to_junction_id((x2, y2), P["junctions"])
        if j1_raw and j2_raw and j1_raw != j2_raw:
            n1, n2 = f"junc:{j1_raw}", f"junc:{j2_raw}"
            if n1 in G and n2 in G:
                if G.has_edge(n1, n2):
                    G[n1][n2]["segments"] = int(G[n1][n2].get("segments", 0)) + 1
                else:
                    G.add_edge(n1, n2, kind="segment", segments=1)

    log.info(f"[graph.build] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G

def assign_net_ids(G: nx.Graph) -> None:
    for i, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            G.nodes[n]["net_id"] = i
