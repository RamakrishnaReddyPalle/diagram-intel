# unified factory using configs/models.yaml
from .clients.qwen2vl import Qwen2VL
from .clients.minicpmv2_6 import MiniCPMV
# add others similarly

def get_client(name:str, cfg):
    if name == "qwen2_vl": return Qwen2VL(cfg)
    if name == "minicpm_v2_6": return MiniCPMV(cfg)
    raise ValueError(f"unknown VLM {name}")
