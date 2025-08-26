from transformers import AutoProcessor, AutoModelForCausalLM
import torch, json, os

class Qwen2VL:
    def __init__(self, cfg):
        mcfg = cfg.vlm.qwen2_vl
        self.device = cfg.env.get("DEVICE","cpu")
        self.dtype  = torch.float16 if cfg.env.get("PRECISION","float16")=="float16" else torch.float32
        self.processor = AutoProcessor.from_pretrained(mcfg.local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            mcfg.local_path, torch_dtype=self.dtype, device_map="auto", trust_remote_code=True
        )

    def ask_json(self, image, prompt:str, schema_hint:str)->dict:
        # schema_hint: brief instruction to return strict JSON matching our pydantic structure
        msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":f"{prompt}\nReturn JSON only.\n{schema_hint}"}]}]
        inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        inputs = {k:v.to(self.model.device) for k,v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=1024)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start:end+1]) if start!=-1 and end!=-1 else {}
