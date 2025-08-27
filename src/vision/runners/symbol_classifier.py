from pathlib import Path
from typing import List, Dict, Any
import json, math

from PIL import Image
from rapidfuzz import fuzz

from src.config.loader import load_cfg
from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging
from src.resources import load_device_catalog
from src.parsers.svg_parse_text import intersect
from src.schema.types import ComponentCandidate, CandidateAlt

# ---------- VLM bootstrap ----------
def _select_vlm(cfg):
    for key in ["qwen2_vl_2b", "qwen2_vl", "llava_v16_mistral_7b"]:
        if key in cfg.vlm and cfg.vlm[key].get("enabled", False):
            return key, cfg.vlm[key].get("local_path")
    return None, None

def _gen_with_vlm(local_path: str, system_str: str, user_str: str, image_path: str, max_new_tokens=180):
    from transformers import AutoProcessor
    import torch

    # Compatible head selection
    ModelCls = None
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq as ModelCls
        except Exception:
            from transformers import Qwen2VLForConditionalGeneration as ModelCls

    processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
    model = ModelCls.from_pretrained(local_path, device_map=None, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    img = Image.open(image_path).convert("RGB")

    messages = [
        {"role":"system","content":[{"type":"text","text":system_str}]},
        {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":user_str}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    s, e = out.find("{"), out.rfind("}")
    return out[s:e+1] if s!=-1 and e!=-1 else "{}"

# ---------- label harvesting for meso tiles ----------
def _labels_in_tile(vec_page_items: List[Dict[str,Any]], tile_bbox: List[int]) -> List[str]:
    out = []
    for it in vec_page_items:
        bb = it["bbox"]
        if intersect((bb[0],bb[1],bb[2],bb[3]), (tile_bbox[0],tile_bbox[1],tile_bbox[2],tile_bbox[3]), 1.0):
            out.append(it["text"])
    # simple dedup
    uniq = []
    for s in out:
        if not any(fuzz.ratio(s.lower(), t.lower())>=95 for t in uniq):
            uniq.append(s)
    return uniq

def classify_meso_tiles(cfg):
    log = setup_logging(cfg.logging.level)
    if not cfg.symbols.use_vlm_on_meso:
        log.info("[symbols] VLM disabled; nothing to do.")
        return

    tile_index_path = Path(cfg.paths.interim) / "tiles" / "tile_index.json"
    assert tile_index_path.exists(), "tile_index.json missing; run tiler."
    tiles = read_json(tile_index_path)
    meso_tiles = [t for t in tiles if t["scale"]=="meso"]

    vec_root = Path(cfg.paths.processed) / "vector_text"
    vec_index = read_json(vec_root / "index.json")

    # Load device catalog
    catalog = load_device_catalog(cfg.root)
    allowed_ids = [t["id"] for t in catalog["types"]]

    # --- PROMPT PATH RESOLUTION (robust) ---
    prompt_path = Path(getattr(cfg, "prompts", {}).get("symbol_classifier", ""))
    candidates = [
        prompt_path,
        (Path(cfg.root) / prompt_path) if not prompt_path.is_absolute() else prompt_path,
        Path(cfg.root) / "src" / "vision" / "prompts" / "classify_symbol.json",
    ]
    prompt_file = next((p for p in candidates if p and p.exists()), None)
    if not prompt_file:
        raise FileNotFoundError("classify_symbol.json not found. Tried:\n  " + "\n  ".join(str(c) for c in candidates if c))
    prompt = json.loads(prompt_file.read_text(encoding="utf-8"))
    system_tmpl = prompt["system"]
    user_tmpl = prompt["user"]


    # pick VLM
    vlm_name, vlm_path = _select_vlm(cfg)
    assert vlm_path, "No enabled VLM found (qwen2_vl_2b/qwen2_vl/llava…). Enable one in configs/models.yaml."

    # Load per-page vector labels into memory
    vec_cache: Dict[str, Dict[int, List[Dict[str,Any]]]] = {}
    for pdf_name, pages in vec_index.items():
        vec_cache[pdf_name] = {}
        for p in pages:
            vec_cache[pdf_name][int(p["page"])] = read_json(p["path"])

    # Rank meso tiles by number of labels intersecting
    scored = []
    for t in meso_tiles:
        pdf, page = t["pdf"], int(t["page"])
        labels_here = _labels_in_tile(vec_cache.get(pdf,{}).get(page,[]), t["bbox"])
        scored.append((len(labels_here), labels_here, t))
    scored.sort(key=lambda x: x[0], reverse=True)

    # filter by min labels and take top-K
    scored = [s for s in scored if s[0] >= int(cfg.symbols.min_labels_in_tile)]
    scored = scored[: int(cfg.symbols.select_top_by_labels)]

    budget = int(cfg.symbols.max_tiles_vlm)
    results: List[ComponentCandidate] = []
    out_root = Path(cfg.paths.processed) / "components" / "candidates"

    for i, (nlab, labels_here, t) in enumerate(scored):
        if i >= budget:
            break

        # Build prompt strings
        allowed_str = "\n".join(allowed_ids)
        labels_str = ", ".join(labels_here[:30]) if labels_here else "(none)"
        system_str = system_tmpl
        user_str = user_tmpl.replace("{ALLOWED_TYPES}", allowed_str).replace("{NEARBY_LABELS}", labels_str)

        raw_json = _gen_with_vlm(
            local_path=vlm_path,
            system_str=system_str,
            user_str=user_str,
            image_path=t["path"],
            max_new_tokens=int(cfg.symbols.max_new_tokens),
        )

        # parse result json
        try:
            obj = json.loads(raw_json)
        except Exception:
            obj = {"type": None, "confidence": 0.0, "ports_expected": [], "notes": "parse_error", "alternatives": []}

        cand = ComponentCandidate(
            id=f"{t['pdf']}:{t['page']}:meso:r{t['row']:03d}c{t['col']:03d}",
            pdf=t["pdf"], page=int(t["page"]),
            tile_path=t["path"], tile_bbox=t["bbox"],
            type=obj.get("type"), confidence=float(obj.get("confidence") or 0.0),
            ports_expected=obj.get("ports_expected") or [],
            notes=obj.get("notes"),
            alternatives=[CandidateAlt(**a) for a in obj.get("alternatives",[]) if isinstance(a, dict)],
            labels_context=labels_here,
            source_model=vlm_name,
        )

        out_path = out_root / t["pdf"] / f"page-{t['page']}" / f"meso_r{t['row']:03d}_c{t['col']:03d}.json"
        ensure_dir(out_path.parent)
        write_json(json.loads(cand.model_dump_json()), out_path)
        results.append(cand)

    # Write index
    idx = [{
        "pdf": c.pdf, "page": c.page, "id": c.id, "tile": c.tile_path,
        "type": c.type, "conf": c.confidence
    } for c in results]
    idx_path = Path(cfg.paths.processed) / "components" / "candidates.index.json"
    write_json(idx, idx_path)
    log.info(f"[symbols] wrote {len(results)} candidates → {idx_path}")
