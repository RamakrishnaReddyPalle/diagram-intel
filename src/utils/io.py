# read/write json/yaml/graph; caching
from pathlib import Path
import json, yaml

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_json(obj, path):
    p = Path(path); ensure_dir(p.parent); p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_yaml(obj, path):
    p = Path(path); ensure_dir(p.parent); p.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
