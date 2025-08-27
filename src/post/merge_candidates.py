from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging
import math, re

BBox = Tuple[float, float, float, float]

def iou(a:BBox, b:BBox)->float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter <= 0: return 0.0
    A = (a[2]-a[0])*(a[3]-a[1]); B=(b[2]-b[0])*(b[3]-b[1])
    return inter / max(1e-6, (A + B - inter))

def edge_dist(a:BBox, b:BBox) -> float:
    dx = max(0, max(a[0]-b[2], b[0]-a[2]))
    dy = max(0, max(a[1]-b[3], b[1]-a[3]))
    return max(dx, dy)

def bbox_union(a:BBox, b:BBox)->BBox:
    return (min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]))

_ROWCOL_RE = re.compile(r"tile_r(\d+)_c(\d+)", re.IGNORECASE)

def _derive_candidate_json_path(processed_root: str, row: Dict[str,Any]) -> Path | None:
    """
    Map an index row → candidate json path.
    tile path: .../tile_rNNN_cMMM.png
    candidate json: .../meso_rNNN_cMMM.json
    """
    base = Path(processed_root) / "components" / "candidates" / row["pdf"] / f"page-{int(row['page'])}"
    tile_stem = Path(row["tile"]).stem  # e.g., tile_r001_c005
    # preferred mapping: tile_* -> meso_*
    cand_name = tile_stem.replace("tile_", "meso_") + ".json"
    p = base / cand_name
    if p.exists():
        return p

    # fallback: parse r/c via regex and format
    m = _ROWCOL_RE.search(tile_stem)
    if m:
        r = int(m.group(1)); c = int(m.group(2))
        cand_name2 = f"meso_r{r:03d}_c{c:03d}.json"
        p2 = base / cand_name2
        if p2.exists():
            return p2

    # last resort: glob by row/col pattern if present, else give up
    hits = list(base.glob("meso_*.json"))
    return hits[0] if hits else None

def load_candidates(processed_root: str) -> Dict[Tuple[str,int], List[Dict[str,Any]]]:
    idx_path = Path(processed_root) / "components" / "candidates.index.json"
    if not idx_path.exists():
        return {}
    idx = read_json(idx_path)

    groups: Dict[Tuple[str,int], List[Dict[str,Any]]] = {}
    for row in idx:
        p = _derive_candidate_json_path(processed_root, row)
        if not p or not p.exists():
            continue
        c = read_json(p)
        # retain row/col from filename for later IDs (best-effort)
        stem = p.stem  # meso_rNNN_cMMM
        try:
            parts = stem.split("_")
            r = int(parts[1][1:])
            col = int(parts[2][1:])
            c["row"] = r; c["col"] = col
        except Exception:
            pass
        groups.setdefault((row["pdf"], int(row["page"])), []).append(c)
    return groups

def cluster_candidates(cands: List[Dict[str,Any]], iou_th: float, touch_px: float):
    n = len(cands)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    boxes = [tuple(c["tile_bbox"]) for c in cands]
    for i in range(n):
        for j in range(i+1, n):
            if iou(boxes[i], boxes[j]) >= iou_th or edge_dist(boxes[i], boxes[j]) <= touch_px:
                union(i,j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())

def merge_cluster(pdf:str, page:int, cands: List[Dict[str,Any]], idxs: List[int],
                  prefer_higher_conf: bool, union_bbox_flag: bool) -> Dict[str,Any]:
    def key(i):
        ci = cands[i]
        conf = float(ci.get("confidence") or 0.0)
        nlab = len(ci.get("labels_context") or [])
        return (conf, nlab)

    rep_i = max(idxs, key=key) if prefer_higher_conf else idxs[0]
    rep = cands[rep_i]

    # bbox
    bb = tuple(rep["tile_bbox"])
    if union_bbox_flag:
        for i in idxs:
            if i==rep_i: continue
            bb = bbox_union(bb, tuple(cands[i]["tile_bbox"]))

    # labels union
    labels = []
    for i in idxs:
        for s in (cands[i].get("labels_context") or []):
            if s not in labels:
                labels.append(s)

    sources = [cands[i].get("id", f"{pdf}:{page}:{i}") for i in idxs]
    tile_paths = [cands[i]["tile_path"] for i in idxs]

    merged = {
        "id": f"{pdf}:{page}:comp:{rep.get('row','NA')}:{rep.get('col','NA')}:{abs(hash(tuple(idxs)))%10**6}",
        "pdf": pdf,
        "page": page,
        "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
        "type": rep.get("type"),
        "confidence": float(rep.get("confidence") or 0.0),
        "ports_expected": rep.get("ports_expected") or [],
        "notes": rep.get("notes"),
        "labels_context": labels,
        "sources": sources,
        "source_tiles": tile_paths,
        "source_count": len(idxs),
    }
    return merged

def run_merge(cfg):
    log = setup_logging(cfg.logging.level)
    groups = load_candidates(cfg.paths.processed)

    out_root = Path(cfg.paths.processed) / "components" / "merged"
    ensure_dir(out_root)

    merged_index = []
    if not groups:
        log.warning("[merge] no candidates found to merge; writing empty index.")
        write_json([], out_root / "merged.index.json")
        return

    for (pdf,page), cands in groups.items():
        clusters = cluster_candidates(
            cands,
            iou_th=float(cfg.merge.iou_threshold),
            touch_px=float(cfg.merge.touch_px),
        )
        page_out_dir = out_root / pdf / f"page-{page}"
        ensure_dir(page_out_dir)

        for cl in clusters:
            merged = merge_cluster(
                pdf, page, cands, cl,
                prefer_higher_conf=bool(cfg.merge.prefer_higher_conf),
                union_bbox_flag=bool(cfg.merge.union_bbox),
            )
            outp = page_out_dir / f"comp_{merged['id'].split(':')[-1]}.json"
            write_json(merged, outp)
            merged_index.append({
                "pdf": pdf, "page": page, "path": str(outp),
                "type": merged["type"], "conf": merged["confidence"], "n_sources": merged["source_count"]
            })

        log.info(f"[merge] {pdf} page-{page}: {len(cands)} → {len(clusters)} merged components")

    write_json(merged_index, out_root / "merged.index.json")
    log.info(f"[merge] wrote {len(merged_index)} entries → {out_root/'merged.index.json'}")
