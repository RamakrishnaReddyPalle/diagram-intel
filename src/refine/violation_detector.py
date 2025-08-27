# src/refine/violation_detector.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Set
import re
import networkx as nx

from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging

# ----------------------------
# Helpers (pack-driven regexes)
# ----------------------------

def _is_composite_component(a: Dict[str, Any], cfg) -> bool:
    """Heuristic: if a component's labels_context is huge, treat it as composite."""
    ctx = a.get("labels_context") or ""
    if isinstance(ctx, list):
        tokens = sum(len(str(x).split()) for x in ctx)
    else:
        tokens = len(str(ctx).split())
    limit = int(cfg.constraints.components.get("composite_heuristics", {})
                .get("max_labels_tokens_for_device_checks", 40))
    return tokens >= limit


def _compile_pack_regexes(cfg):
    """Compile generic source/load regexes from constraint packs (generic, indian_power, etc.)."""
    sk = (cfg.constraints.inference.get("source_keywords") or {})
    compiled = {}
    for cat, patterns in sk.items():
        try:
            compiled[cat] = [re.compile(p) for p in patterns]
        except Exception:
            compiled[cat] = []
    return compiled

def _net_text_blob(G: nx.Graph, net_id: int) -> str:
    """Aggregate labels_context/type strings for all nodes on a net into one blob."""
    parts: List[str] = []
    for n, a in G.nodes(data=True):
        if a.get("net_id") == net_id:
            ctx = a.get("labels_context")
            if isinstance(ctx, (list, tuple)):
                parts.extend([str(x) for x in ctx])
            elif isinstance(ctx, str):
                parts.append(ctx)
            t = a.get("type")
            if t:
                parts.append(str(t))
    return " | ".join(parts)

def _match_any(regex_list: List[re.Pattern], text: str) -> bool:
    return any(r.search(text) for r in regex_list)

def _changeover_keys_from_hints(cfg) -> List[str]:
    hints = (cfg.constraints.inference.get("typing_hints_contains") or {})
    keys: List[str] = []
    for k, vals in hints.items():
        if k.upper() in {"ACCL", "ATS", "SELECTOR", "THREEWAY"}:
            keys.extend([v.lower() for v in vals])
    # Always include a few generic substrings to be safe
    keys.extend(["changeover", "selector", "3way", "threeway"])
    # dedupe
    keys = sorted(set(keys))
    return keys

def _has_changeover_on_net(G: nx.Graph, net_id: int, cfg) -> bool:
    keys = _changeover_keys_from_hints(cfg)
    for n, a in G.nodes(data=True):
        if a.get("net_id") != net_id:
            continue
        text = a.get("labels_context") or ""
        if isinstance(text, list):
            text = " | ".join([str(x) for x in text])
        text_low = str(text).lower()
        if any(k in text_low for k in keys):
            return True
        t = (a.get("type") or "")
        if any(k in str(t).lower() for k in keys):
            return True
    return False

def _looks_like_rccb(a: Dict[str, Any]) -> bool:
    ctx = a.get("labels_context") or ""
    if isinstance(ctx, list):
        ctx = " | ".join([str(x) for x in ctx])
    ctx_up = str(ctx).upper()
    t = str(a.get("type") or "").upper()
    return ("RCCB" in ctx_up) or ("RCD" in ctx_up) or ("ELCB" in ctx_up) or \
           ("RCCB" in t) or ("RCD" in t) or ("ELCB" in t)

# ----------------------------
# Graph loading
# ----------------------------

def _choose_graph_path(cfg, pdf_stem: str, page: int) -> Path:
    """Prefer refined graph if present, else the base stitched graph."""
    base = Path(cfg.paths.processed) / "graphs" / pdf_stem / f"page-{page}.json"
    refined = Path(cfg.paths.processed) / "graphs_refined" / pdf_stem / f"page-{page}.json"
    return refined if refined.exists() else base

def _load_graph(cfg, pdf_stem: str, page: int) -> nx.Graph:
    gpath = _choose_graph_path(cfg, pdf_stem, page)
    payload = read_json(gpath)
    G = nx.Graph()
    # nodes
    for nd in payload.get("nodes", []):
        nid = nd.get("id")
        attrs = {k: v for k, v in nd.items() if k != "id"}
        G.add_node(nid, **attrs)
    # edges
    for ed in payload.get("edges", []):
        u, v = ed.get("u"), ed.get("v")
        attrs = {k: v for k, v in ed.items() if k not in ("u", "v")}
        if u is not None and v is not None:
            G.add_edge(u, v, **attrs)
    # keep some graph attrs (optional)
    for k, v in payload.get("graph_attrs", {}).items():
        G.graph[k] = v
    return G

# ----------------------------
# Main detection routine
# ----------------------------

def detect_violations(cfg, pdf_stem: str, page: int = 1) -> Dict[str, Any]:
    """
    Run portable checks:
      - giant-net thresholds
      - source-bridge-without-changeover (pack-driven)
      - RCCB-no-isolation (neighbors all on same net)
    Write processed/refine/<pdf>/page-<page>.violations.json and return payload.
    """
    log = setup_logging(cfg.logging.level)

    G = _load_graph(cfg, pdf_stem, page)
    violations: List[Dict[str, Any]] = []

    # 1) Giant net checks
    max_warn = int(cfg.constraints.nets.get("max_nodes_warning", 2500))
    max_err  = int(cfg.constraints.nets.get("max_nodes_error", 15000))
    net_sizes: Dict[int, int] = {}
    for _, a in G.nodes(data=True):
        nid = a.get("net_id")
        if nid is None:
            continue
        net_sizes[nid] = net_sizes.get(nid, 0) + 1

    for nid, sz in sorted(net_sizes.items(), key=lambda x: -x[1]):
        if sz >= max_err:
            violations.append({
                "type": "giant_net",
                "severity": "error",
                "net_id": nid,
                "size": sz,
                "limit": max_err,
                "message": f"Net {nid} has {sz} nodes (≥ error limit {max_err})."
            })
        elif sz >= max_warn:
            violations.append({
                "type": "giant_net",
                "severity": "warning",
                "net_id": nid,
                "size": sz,
                "limit": max_warn,
                "message": f"Net {nid} has {sz} nodes (≥ warning limit {max_warn})."
            })

    # 2) Source bridge without changeover (pack-driven)
    src_regex = _compile_pack_regexes(cfg)
    categories = list(src_regex.keys())

    # Precompute net blobs
    net_blobs: Dict[int, str] = {}
    for nid in net_sizes.keys():
        net_blobs[nid] = _net_text_blob(G, nid)

    for nid, blob in net_blobs.items():
        present = {cat for cat in categories if _match_any(src_regex.get(cat, []), blob)}
        if len(present) >= 2:
            if not _has_changeover_on_net(G, nid, cfg):
                violations.append({
                    "type": "source_bridge_without_changeover",
                    "severity": "warning",
                    "net_id": nid,
                    "sources_detected": sorted(list(present)),
                    "message": f"Net {nid} shows multiple source signatures {sorted(list(present))} without a changeover/ATS."
                })

    # 3) RCCB not isolating (all neighbor nets identical)
    for n, a in G.nodes(data=True):
        if a.get("kind") != "component":
            continue
        if _is_composite_component(a, cfg):
            continue  # skip mega-components (likely merged regions/title blocks)
        if not _looks_like_rccb(a):
            continue
        nbr_net_ids: Set[int] = {
            G.nodes[nbr].get("net_id") for nbr in G.neighbors(n)
            if G.nodes[nbr].get("net_id") is not None
        }
        if len(nbr_net_ids) <= 1:
            violations.append({
                "type": "rccb_no_isolation",
                "severity": "warning",
                "node": n,
                "neighbor_nets": sorted(list(nbr_net_ids)),
                "message": "RCCB appears not to isolate (all neighbors share one net)."
            })


    # Pack results
    payload = {
        "pdf": pdf_stem,
        "page": page,
        "stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "nets_count": len(net_sizes),
            "largest_net": max(net_sizes.values()) if net_sizes else 0,
        },
        "violations": violations,
    }

    out_dir = Path(cfg.paths.processed) / "refine" / pdf_stem
    ensure_dir(out_dir)
    out_path = out_dir / f"page-{page}.violations.json"
    write_json(payload, out_path)
    log.info(f"[refine] wrote violations → {out_path}")
    return payload
