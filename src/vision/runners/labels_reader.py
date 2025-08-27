# src/vision/runners/labels_reader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from rapidfuzz import fuzz

from src.utils.io import write_json, read_json, ensure_dir
from src.utils.logging import setup_logging
from src.parsers.svg_parse_text import parse_pdf_text_fitz, parse_svg_text, intersect

# --- tiny VLM utility (Qwen2-VL preferred) ---
def _select_vlm(cfg):
    # Priority: qwen2_vl_2b -> qwen2_vl -> llava_v16_mistral_7b
    vlms = getattr(cfg, "vlm", {}) or {}
    for key in ["qwen2_vl_2b", "qwen2_vl", "llava_v16_mistral_7b"]:
        meta = vlms.get(key)
        if meta and meta.get("enabled", False) and meta.get("local_path"):
            return key, meta["local_path"]
    return None, None

def _vlm_labels_from_image(local_path: str, img_path: str, max_new_tokens=128) -> List[str]:
    """Ask the VLM to output JSON: {"labels": ["...","..."]}"""
    from transformers import AutoProcessor
    import torch, json as pyjson

    processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)

    # model class fallback chain
    ModelCls = None
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq as ModelCls
        except Exception:
            from transformers import Qwen2VLForConditionalGeneration as ModelCls

    model = ModelCls.from_pretrained(local_path, torch_dtype=torch.float32, device_map=None, trust_remote_code=True)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    prompt = (
        'Extract short textual labels visible in this image (device names, port tags, phases, ratings). '
        'Return only strict JSON like {"labels":["L1","L2","MCCB","TPN 125A"]} with no extra text.'
    )
    messages = [{
        "role":"user",
        "content":[{"type":"image","image":img},{"type":"text","text":prompt}],
    }]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    s = out.find("{"); e = out.rfind("}")
    if s!=-1 and e!=-1:
        try:
            obj = pyjson.loads(out[s:e+1])
            labels = obj.get("labels", [])
            labels = [str(t).strip() for t in labels if str(t).strip()]
            return labels
        except Exception:
            pass
    return []

# --- OCR fallback (Donut base; best-effort) ---
def _select_ocr_model_path(cfg) -> str | None:
    """
    Try cfg.models.donut.local_path, else models/registry.json['donut'].local_path
    """
    # 1) configs
    try:
        p = cfg.models.donut.local_path
        if p and Path(p).exists():
            return str(p)
    except Exception:
        pass
    # 2) registry.json
    reg = Path(getattr(cfg.paths, "model_cache", "./models/cache")).parents[0] / "registry.json"
    if reg.exists():
        try:
            import json
            j = json.loads(reg.read_text(encoding="utf-8"))
            p = j.get("donut", {}).get("local_path")
            if p and Path(p).exists():
                return p
        except Exception:
            pass
    return None

def _ocr_labels_from_image(local_path: str, img_path: str, max_new_tokens=64) -> List[str]:
    """
    Very lightweight OCR-ish fallback using Donut base.
    We do NOT assume a dataset-specific prompt; just ask for words we can read.
    """
    try:
        from transformers import AutoProcessor, VisionEncoderDecoderModel
        import torch, re
        processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
        model = VisionEncoderDecoderModel.from_pretrained(local_path, trust_remote_code=True)
        model.eval()

        img = Image.open(img_path).convert("RGB")
        # crude: treat this as captioning -> split tokens
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        with torch.no_grad():
            out_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # tokenize to candidate "labels"
        words = [w.strip(",.;:()[]{}") for w in re.split(r"[\s/|]+", text)]
        words = [w for w in words if w and len(w) >= 2]
        # keep alnum & -+% only
        words = [w for w in words if re.fullmatch(r"[A-Za-z0-9\-+%]+", w)]
        # simple post rules to capture common tags like TB1-4, L1/L2, 100A
        return list(dict.fromkeys(words))[:30]
    except Exception:
        return []

# --- main vector text pass ---
def build_vector_text_index(cfg) -> Dict[str, Any]:
    """
    For each PDF/page, try SVG text first; if empty, fall back to PyMuPDF text.
    Dump per-page JSON to processed/vector_text/<pdf>/page-#.json
    """
    log = setup_logging(cfg.logging.level)
    raw_manifests = Path(cfg.paths.raw) / "manifests"
    out_root = Path(cfg.paths.processed) / "vector_text"
    ensure_dir(out_root)

    index = {}
    for manifest_path in sorted(raw_manifests.glob("*.json")):
        m = read_json(manifest_path)
        pdf_name = Path(m["pdf"]).stem
        pdf_path = m["pdf"]
        idx_pages = []

        for meta in m["pages"]:
            page = int(meta["page"])
            svg_path = meta.get("svg")
            items = []
            src = None

            if svg_path and Path(svg_path).exists():
                items = parse_svg_text(svg_path, min_chars=cfg.labels.min_vec_chars)
                src = "svg"

            if not items:
                items = parse_pdf_text_fitz(pdf_path, page, dpi=cfg.runtime.dpi, min_chars=cfg.labels.min_vec_chars)
                src = "pymupdf"

            page_out = out_root / pdf_name / f"page-{page}.json"
            write_json(items, page_out)
            idx_pages.append({"page": page, "path": str(page_out), "count": len(items), "source": src})

        index[pdf_name] = idx_pages
        log.info(f"[vector_text] {pdf_name}: "
                 f"{sum(p['count'] for p in idx_pages)} items across {len(idx_pages)} pages "
                 f"(sources: {[p['source'] for p in idx_pages]})")

    write_json(index, out_root / "index.json")
    return index

# --- tile mapping + optional VLM/OCR augmentation ---
def run_labels_reader(cfg):
    """
    Build per-tile labels:
      - collect all vector text intersecting the tile bbox
      - *fallbacks*:
          * if page vector text is scarce and cfg.ocr.enable -> OCR tiles (Donut)
          * else if cfg.labels.use_vlm_on_micro -> VLM per tile (capped)
      - merge & deduplicate (fuzzy)
    """
    log = setup_logging(cfg.logging.level)
    tile_index_path = Path(cfg.paths.interim) / "tiles" / "tile_index.json"
    assert tile_index_path.exists(), "tile_index.json not found. Run tiler first."

    vec_root = Path(cfg.paths.processed) / "vector_text"
    assert (vec_root / "index.json").exists(), "Vector text index missing. Run build_vector_text_index first."

    vec_idx = read_json(vec_root / "index.json")

    # budgets and toggles
    vlm_name, vlm_path = _select_vlm(cfg)
    use_vlm = bool(getattr(cfg.labels, "use_vlm_on_micro", False) and vlm_path)

    ocr_enabled = bool(getattr(cfg, "ocr", {}).get("enable", False))
    min_vec_threshold = int(getattr(cfg, "ocr", {}).get("min_vec_text_threshold", 20))
    ocr_model_path = _select_ocr_model_path(cfg) if ocr_enabled else None

    if use_vlm:
        log.info(f"[labels] VLM enabled for micro tiles: {vlm_name}")
    else:
        log.info("[labels] VLM disabled for micro tiles.")

    if ocr_enabled and ocr_model_path:
        log.info(f"[labels] OCR fallback enabled (Donut) with min_vec_text_threshold={min_vec_threshold}")
    elif ocr_enabled:
        log.warning("[labels] OCR enabled but Donut local path not found; skipping OCR fallback.")

    tiles = read_json(tile_index_path)
    micro_tiles = [t for t in tiles if t["scale"] == "micro"]

    out_root_tiles = Path(cfg.paths.processed) / "labels" / "tiles"
    out_root_tiles.mkdir(parents=True, exist_ok=True)

    # Load all per-page vector text into memory (per pdf)
    cache_vec: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    page_vec_counts: Dict[str, Dict[int, int]] = {}
    for pdf_name, pages in vec_idx.items():
        cache_vec[pdf_name] = {}
        page_vec_counts[pdf_name] = {}
        for pinfo in pages:
            pg = int(pinfo["page"])
            cache_vec[pdf_name][pg] = read_json(pinfo["path"])
            page_vec_counts[pdf_name][pg] = int(pinfo.get("count", len(cache_vec[pdf_name][pg])))

    def merge_dedup(vec_labels: List[str], fx_labels: List[str]) -> List[str]:
        out = list(vec_labels)
        for cand in fx_labels:
            if not cand:
                continue
            exists = any(fuzz.ratio(cand.lower(), v.lower()) >= getattr(cfg.labels, "merge_dedup_fuzz", 88) for v in out)
            if not exists:
                out.append(cand)
        return out

    vlm_budget = int(getattr(cfg.labels, "max_tiles_vlm", 120))
    vlm_used = 0
    per_tile_records = []

    for t in micro_tiles:
        pdf = t["pdf"]; page = int(t["page"]); bbox = t["bbox"]
        vec_items = cache_vec.get(pdf, {}).get(page, [])
        vec_in_tile = []
        for it in vec_items:
            bb = it["bbox"]
            if intersect((bb[0],bb[1],bb[2],bb[3]), (bbox[0],bbox[1],bbox[2],bbox[3]), min_overlap_px=1.0):
                vec_in_tile.append(it)
        vec_labels = [it["text"] for it in vec_in_tile]

        # decide fallback per page/tile
        fallback_labels: List[str] = []
        scarce_page_vec = page_vec_counts.get(pdf, {}).get(page, 0) < min_vec_threshold
        if ocr_enabled and ocr_model_path and (scarce_page_vec or len(vec_labels) == 0):
            fallback_labels = _ocr_labels_from_image(ocr_model_path, t["path"])
        elif use_vlm and (vlm_used < vlm_budget) and len(vec_labels) == 0:
            fallback_labels = _vlm_labels_from_image(vlm_path, t["path"])
            vlm_used += 1

        merged = merge_dedup(vec_labels, fallback_labels)

        # Write per-tile json
        tile_out = out_root_tiles / pdf / f"page-{page}" / Path(t["path"]).name.replace(".png",".json")
        ensure_dir(tile_out.parent)
        rec = {
            "pdf": pdf,
            "page": page,
            "scale": t["scale"],
            "row": t["row"], "col": t["col"],
            "tile_bbox": t["bbox"],
            "tile_path": t["path"],
            "vector_labels": vec_labels,
            "fallback_labels": fallback_labels,
            "labels_merged": merged,
            "vector_items": vec_in_tile,
        }
        write_json(rec, tile_out)
        per_tile_records.append({"tile_json": str(tile_out), "n_vec": len(vec_labels), "n_fallback": len(fallback_labels)})

    out_idx = Path(cfg.paths.processed) / "labels" / "tile_labels.index.json"
    write_json(per_tile_records, out_idx)
    log.info(f"[labels] wrote {len(per_tile_records)} tile label files → {out_idx}")
