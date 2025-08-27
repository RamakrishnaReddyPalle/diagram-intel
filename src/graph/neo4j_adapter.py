# src/graph/neo4j_adapter.py
from __future__ import annotations
from typing import Dict, Any
from neo4j import GraphDatabase
import json

def _flat(v):
    # convert lists/dicts/None -> strings for Neo4j properties
    if v is None: return ""
    if isinstance(v, (list, dict, set, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return v

def push_graph(cfg, pdf_stem: str, page: int, G) -> Dict[str, Any]:
    neo = cfg.get("neo4j", {}) or cfg.neo4j
    if not getattr(neo, "enabled", False):
        return {"pushed": False, "reason": "neo4j.disabled"}

    driver = GraphDatabase.driver(neo.uri, auth=(neo.user, neo.password))
    with driver.session(database=neo.database) as session:
        session.execute_write(_upsert_graph_tx, pdf_stem, page, G)
    driver.close()
    return {"pushed": True, "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

def _upsert_graph_tx(tx, pdf_stem, page, G):
    # label sets
    for n, a in G.nodes(data=True):
        kind = (a.get("kind") or "Node").capitalize()
        props = {k: _flat(v) for k, v in a.items()}
        props["id"] = n
        props["pdf_stem"] = pdf_stem
        props["page"] = page
        tx.run(f"""
            MERGE (x:{kind} {{id:$id}})
            SET x += $props
        """, id=n, props=props)

    for u, v, a in G.edges(data=True):
        props = {k: _flat(vv) for k, vv in a.items()}
        props["pdf_stem"] = pdf_stem
        props["page"] = page
        tx.run("""
            MATCH (a {id:$u}), (b {id:$v})
            MERGE (a)-[r:LINKS_TO]->(b)
            SET r += $props
        """, u=u, v=v, props=props)
