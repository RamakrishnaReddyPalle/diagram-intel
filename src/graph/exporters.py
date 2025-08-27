from pathlib import Path
import networkx as nx
from src.utils.io import write_json, ensure_dir
from src.utils.logging import setup_logging

def _clean_val(v):
    # GraphML supports str/int/float/bool — normalize everything else.
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return v
    return str(v)

def _clean_attrs(d: dict) -> dict:
    return {k: _clean_val(v) for k, v in d.items()}

def export_graph(cfg, pdf_stem: str, page: int, G: nx.Graph):
    log = setup_logging(cfg.logging.level)
    out_dir = Path(cfg.paths.processed) / "graphs" / pdf_stem
    ensure_dir(out_dir)

    # 1) Always write JSON first (robust)
    payload = {
        "graph_attrs": G.graph,
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes()],
        "edges": [{"u": u, "v": v, **G.edges[u, v]} for u, v in G.edges()],
    }
    write_json(payload, out_dir / f"page-{page}.json")

    # 2) GraphML (optional, sanitized)
    if bool(cfg.graph.export.write_graphml):
        try:
            G2 = nx.Graph(**_clean_attrs(G.graph))
            for n, attrs in G.nodes(data=True):
                G2.add_node(n, **_clean_attrs(attrs))
            for u, v, attrs in G.edges(data=True):
                G2.add_edge(u, v, **_clean_attrs(attrs))
            nx.write_graphml(G2, out_dir / f"page-{page}.graphml")
        except Exception as e:
            log.warning(f"[graph.export] GraphML export skipped: {e}. "
                        f"JSON was written to {out_dir / f'page-{page}.json'}")
