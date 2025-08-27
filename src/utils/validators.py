# schema validators
# src/utils/validators.py
from __future__ import annotations
from typing import Dict, Any, List

def nonempty_str(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def validate_component(c: Dict[str, Any]) -> List[str]:
    err=[]
    for k in ["id","bbox"]:
        if k not in c: err.append(f"missing {k}")
    if "bbox" in c:
        bb = c["bbox"]
        if not (isinstance(bb,(list,tuple)) and len(bb)==4):
            err.append("bbox must be [x1,y1,x2,y2]")
    return err

def validate_port(p: Dict[str, Any]) -> List[str]:
    err=[]
    for k in ["id","xy"]:
        if k not in p: err.append(f"missing {k}")
    return err

def validate_net(n: Dict[str, Any]) -> List[str]:
    err=[]
    if "net_id" not in n: err.append("missing net_id")
    return err
