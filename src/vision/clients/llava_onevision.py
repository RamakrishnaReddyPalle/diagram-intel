# src/vision/clients/llava_onevision.py
from __future__ import annotations
from typing import Dict, Any, List
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

def llava_generate_json(local_path: str, img_path: str, prompt: str, max_new_tokens=256) -> str:
    processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(local_path, torch_dtype=torch.float32, device_map=None, trust_remote_code=True)
    model.eval()
    img = Image.open(img_path).convert("RGB")
    messages=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
