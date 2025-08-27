# src/vision/clients/qwen2vl.py
from __future__ import annotations
from transformers import AutoProcessor
from PIL import Image
import torch, json

def _pick_model_class():
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
        return ModelCls
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq as ModelCls
            return ModelCls
        except Exception:
            from transformers import Qwen2VLForConditionalGeneration as ModelCls
            return ModelCls

def _get_local_path(cfg):
    # prefer 2B if present, else 7B
    try:
        return cfg.models.qwen2_vl_2b.local_path
    except Exception:
        return cfg.models.qwen2_vl.local_path

def _dtype_from_cfg(cfg):
    prec = str(getattr(cfg.env, "PRECISION", "float32")).lower()
    return torch.float16 if "16" in prec else torch.float32

def _build_io(local_path: str):
    processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
    ModelCls = _pick_model_class()
    model = ModelCls.from_pretrained(local_path, torch_dtype=torch.float32, device_map=None, trust_remote_code=True)
    model.eval()
    return processor, model

def _gen_text_only(processor, model, user_text: str, max_new_tokens=128):
    messages=[{"role":"user","content":[{"type":"text","text":user_text}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# ------------ public helpers ------------

def qwen_table_json(cfg, img_path: str, max_new_tokens=256):
    local = _get_local_path(cfg)
    processor, model = _build_io(local)
    img = Image.open(img_path).convert("RGB")
    prompt = 'Extract table to strict JSON {"rows":[["c1","c2",...],...]} — return JSON only.'
    messages=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    s = out.find("{"); e = out.rfind("}")
    if s != -1 and e != -1:
        try:
            obj = json.loads(out[s:e+1])
            return obj.get("rows")
        except Exception:
            return None
    return None

def qwen_summarize_component(cfg, info: dict, max_new_tokens=128) -> str:
    local = _get_local_path(cfg)
    processor, model = _build_io(local)
    tmpl = (cfg.prompts.summarize_component if hasattr(cfg, "prompts") else
            "TYPE={{type}} PHASE={{net_phase}} VOLT={{net_voltage}} LABELS={{labels_context}}")
    user = tmpl
    for k, v in info.items():
        user = user.replace("{{"+k+"}}", str(v))
    out = _gen_text_only(processor, model, user, max_new_tokens=max_new_tokens)
    return out.strip()

# optional thin OO wrapper if you want it
class Qwen2VL:
    def __init__(self, cfg):
        local = _get_local_path(cfg)
        self.processor, self.model = _build_io(local)

    def ask_json(self, image: Image.Image, prompt: str, schema_hint: str, max_new_tokens=256) -> dict:
        messages = [{"role":"user","content":[
            {"type":"image","image":image},
            {"type":"text","text":f"{prompt}\nReturn strict JSON only.\n{schema_hint}"}
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        s = out.find("{"); e = out.rfind("}")
        try:
            return json.loads(out[s:e+1]) if s!=-1 and e!=-1 else {}
        except Exception:
            return {}
