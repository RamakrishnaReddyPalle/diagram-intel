from pathlib import Path
from typing import List, Dict, Any, Tuple
import math
from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging

BBox = Tuple[float,float,float,float]

def _pt_rect_dist(px, py, bb:BBox) -> float:
    x1,y1,x2,y2 = bb
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return max(dx, dy)  # L-inf distance (good for snap box)

def _which_side(px,py, bb:BBox) -> str:
    x1,y1,x2,y2 = bb
    cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
    dx = px - cx; dy = py - cy
    # decide by which edge is nearest
    dists = {
        "left":  abs(px - x1),
        "right": abs(px - x2),
        "top":   abs(py - y1),
        "bottom":abs(py - y2),
    }
    return min(dists, key=dists.get)

def _cluster_points(pts: List[Tuple[int,int]], radius: float) -> List[List[Tuple[int,int]]]:
    # simple union-find by radius
    n = len(pts)
    if n==0: return []
    parent = list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra
    for i in range(n):
        x1,y1 = pts[i]
        for j in range(i+1,n):
            x2,y2 = pts[j]
            if max(abs(x1-x2), abs(y1-y2)) <= radius:
                union(i,j)
    clusters={}
    for i in range(n):
        r=find(i); clusters.setdefault(r,[]).append(pts[i])
    return list(clusters.values())

def snap_wires_to_components(cfg, pdf_stem: str):
    log = setup_logging(cfg.logging.level)
    wires_path = Path(cfg.paths.processed)/"wires"/pdf_stem/"page-1.json"  # single-page for now
    assert wires_path.exists(), f"wires not found: {wires_path}"
    wires = read_json(wires_path)

    merged_idx = Path(cfg.paths.processed)/"components"/"merged"/"merged.index.json"
    assert merged_idx.exists(), "merged components index missing; run merge step."
    merged_list = read_json(merged_idx)
    page_recs = [r for r in merged_list if r["pdf"]==pdf_stem and int(r["page"])==int(wires["page"])]
    comps = [read_json(Path(r["path"])) for r in page_recs]  # each has bbox, labels_context, etc.

    snap_px = float(cfg.geometry.snap.snap_px)
    junc_px = float(cfg.geometry.snap.junction_px)

    # 1) cluster endpoints into junctions (T-joints, etc.)
    endpoints = [tuple(pt) for pt in wires["endpoints"]]
    junction_sets = _cluster_points(endpoints, junc_px)
    junctions = []
    for i, cluster in enumerate(junction_sets):
        # centroid
        cx = sum(p[0] for p in cluster)/len(cluster); cy = sum(p[1] for p in cluster)/len(cluster)
        junctions.append({"id": f"J{i:04d}", "xy": [round(cx,1), round(cy,1)], "members": cluster})

    # 2) snap endpoints to nearest component bbox
    connections = []  # wire_endpoint -> (comp, port)
    ports = []        # created anchors on components

    # temp id alloc per component
    comp_port_counters = {c["id"]: 0 for c in comps}

    # helper to allocate a port id on a component
    def _alloc_port_id(cid:str)->str:
        comp_port_counters[cid]+=1
        return f"P{comp_port_counters[cid]:02d}"

    for ep in endpoints:
        ex,ey = ep
        # nearest component within snap_px
        best = (None, 1e9, None)  # (comp, dist, side)
        for c in comps:
            bb = tuple(c["bbox"])
            d = _pt_rect_dist(ex,ey,bb)
            if d <= snap_px and d < best[1]:
                best = (c, d, _which_side(ex,ey,bb))
        if best[0] is None:
            continue
        comp = best[0]; side=best[2]
        # is there already a port near this endpoint? (avoid duplicates)
        found = None
        for p in ports:
            if p["comp_id"]==comp["id"] and max(abs(p["xy"][0]-ex), abs(p["xy"][1]-ey))<=8:
                found = p; break
        if found is None:
            pid = _alloc_port_id(comp["id"])
            p = {"comp_id": comp["id"], "port_id": pid, "xy":[int(ex),int(ey)], "side": side}
            ports.append(p)
        else:
            pid = found["port_id"]

        connections.append({
            "endpoint": [int(ex),int(ey)],
            "comp_id": comp["id"],
            "port_id": pid
        })

    out = {
        "pdf": pdf_stem,
        "page": int(wires["page"]),
        "junctions": junctions,
        "ports": ports,
        "connections": connections,
        "wires_ref": str(wires_path),
        "components_ref": [r["path"] for r in page_recs]
    }
    out_path = Path(cfg.paths.processed)/"ports"/pdf_stem/f"page-{wires['page']}.json"
    ensure_dir(out_path.parent)
    write_json(out, out_path)
    log.info(f"[ports] {pdf_stem} page-{wires['page']}: ports={len(ports)} conns={len(connections)} → {out_path}")
