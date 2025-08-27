from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging

# optional VLM helper (Qwen2-VL)
def _try_qwen_table_json(cfg, img_path: str, max_new_tokens: int = 256):
    try:
        from src.vision.clients.qwen2vl import qwen_table_json
        return qwen_table_json(cfg, img_path, max_new_tokens=max_new_tokens)
    except Exception:
        return None

def run_table_reader(
    cfg,
    pdf_stem: str,
    page: int = 1,
    top_k_tiles: int = 8,
    min_rows: int = 2,
    min_cols: int = 2,
) -> Dict[str, Any]:
    """
    Heuristic: pick the micro tiles with the most labels (vector+vlm),
    ask Qwen2-VL to extract tables from those crops, and write a single
    JSON file under processed/tables/<pdf>/page-<page>.json.
    """
    log = setup_logging(cfg.logging.level)

    idx_path = Path(cfg.paths.processed) / "labels" / "tile_labels.index.json"
    if not idx_path.exists():
        raise FileNotFoundError("tile_labels.index.json missing; run labels_reader first.")

    idx = read_json(idx_path)
    # focus this pdf+page, sort by texty-ness
    cands = [
        r for r in idx
        if (pdf_stem in r["tile_json"]) and (f"page-{page}" in r["tile_json"])
    ]
    cands.sort(key=lambda r: (r.get("n_vec", 0) + r.get("n_vlm", 0)), reverse=True)
    cands = cands[:top_k_tiles]

    tables: List[Dict[str, Any]] = []
    for r in cands:
        rec = read_json(r["tile_json"])
        img_path = rec.get("tile_path")
        if not img_path or not Path(img_path).exists():
            continue
        rows = _try_qwen_table_json(cfg, img_path, max_new_tokens=getattr(cfg.tables, "max_new_tokens", 256))
        if not rows or not isinstance(rows, list):
            continue
        # simple validity check
        width_guess = max((len(x) if isinstance(x, list) else 0) for x in rows) if rows else 0
        if len(rows) >= min_rows and width_guess >= min_cols:
            tables.append({
                "tile_json": r["tile_json"],
                "tile_path": img_path,
                "rows": rows,
            })

    out_dir = Path(cfg.paths.processed) / "tables" / pdf_stem
    ensure_dir(out_dir)
    out_path = out_dir / f"page-{page}.json"
    write_json({"pdf": pdf_stem, "page": page, "count": len(tables), "tables": tables}, out_path)
    log.info(f"[table_reader] found {len(tables)} tables → {out_path}")
    return {"count": len(tables), "path": str(out_path)}
