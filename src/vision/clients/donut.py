# src/vision/clients/donut.py
from __future__ import annotations
from typing import List
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel
import torch, re

def donut_read_table(cfg, image_path: str, max_new_tokens=256) -> List[List[str]] | None:
    """Very rough 'table' via text tokens; returns rows split heuristically."""
    try:
        local = cfg.models.donut.local_path
    except Exception:
        return None
    if not local: return None

    processor = AutoProcessor.from_pretrained(local, trust_remote_code=True)
    model = VisionEncoderDecoderModel.from_pretrained(local, trust_remote_code=True)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        out_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    # Split lines -> cells; crude but works for legend blocks with separators.
    lines = [l.strip() for l in re.split(r"[\\n\\r]+", text) if l.strip()]
    rows = []
    for l in lines:
        cells = [c.strip() for c in re.split(r"[\\s\\|]{2,}|\\t|\\s{4,}", l) if c.strip()]
        if cells:
            rows.append(cells)
    return rows if rows else None
