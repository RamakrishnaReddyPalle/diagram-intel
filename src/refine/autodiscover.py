from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from collections import Counter, defaultdict
from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging

TOKEN = re.compile(r"[A-Za-z0-9\-\_/\.]+")  # simple tokeniser for labels

ANCHORS_SRC = re.compile(r"(?i)\b(from|input|incoming|incomer|supply|mains|utility)\b")
ANCHORS_LOAD = re.compile(r"(?i)\b(to|out|output|feeder|outgoing)\b")

def _ngramize(tokens: List[str], n: int = 2) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def discover_constraints_candidates(cfg, pdf_stem: str, page_range: List[int] = None,
                                    top_k: int = 30) -> Dict[str, Any]:
    log = setup_logging(cfg.logging.level)
    vec_dir = Path(cfg.paths.processed)/"vector_text"/pdf_stem
    pages = page_range or sorted({int(p.stem.split("-")[1]) for p in vec_dir.glob("page-*.json")})

    src_counts = Counter()
    load_counts = Counter()
    device_counts = Counter()

    # broad device keys we care about for typing hints
    device_clues = ["MCCB","MCB","RCCB","RCD","ELCB","Isolator","TPN","SPD","ACCL","ATS","Selector","TB","CTS"]

    for pg in pages:
        items = read_json(vec_dir/f"page-{pg}.json")
        # Flatten line order by Y then X (rough reading order)
        items_sorted = sorted(items, key=lambda t: (t["y"], t["x"]))
        lines = [i["text"] for i in items_sorted]
        text = " | ".join(lines)

        # collect tokens
        toks = TOKEN.findall(text)
        toks_lower = [t.lower() for t in toks]

        # device clues (case-sensitive pass over original tokens)
        for t in toks:
            for key in device_clues:
                if key.lower() in t.lower():
                    device_counts[key] += 1

        # anchor windows: take +/- 5 tokens around matches
        for m in ANCHORS_SRC.finditer(text):
            start = max(0, m.start()-80); end = min(len(text), m.end()+80)
            window = TOKEN.findall(text[start:end])
            for n in [1,2,3]:
                for g in _ngramize(window, n):
                    src_counts[g] += 1

        for m in ANCHORS_LOAD.finditer(text):
            start = max(0, m.start()-80); end = min(len(text), m.end()+80)
            window = TOKEN.findall(text[start:end])
            for n in [1,2,3]:
                for g in _ngramize(window, n):
                    load_counts[g] += 1

    # filter out pure numbers and short junk
    def _clean(counter: Counter) -> List[str]:
        out = []
        for k, _ in counter.most_common():
            if len(k) < 2: continue
            if re.fullmatch(r"[0-9\-_/\.]+", k): continue
            out.append(k)
        return out

    src_top  = _clean(src_counts)[:top_k]
    load_top = _clean(load_counts)[:top_k]
    dev_top  = [k for k,_ in device_counts.most_common()]

    # build YAML-like dict
    suggestion = {
        "inference": {
            "source_keywords": {
                "grid": [],
                "dg": [],
                "pv": [],
                "ups": [],
                # We’ll put generic source window results here;
                # you can manually move items into specific buckets later.
                "generic_source_phrases": src_top,
            },
            "load_keywords": load_top,
            "typing_hints_contains": {k: [k] for k in dev_top}  # minimal seed
        }
    }

    out_dir = Path(cfg.root)/"configs"/"constraints"/"projects"
    ensure_dir(out_dir)
    out_path = out_dir/f"suggested_{pdf_stem}.yaml"
    # write as JSON for reliability; you can rename to .yaml, it’s compatible
    write_json(suggestion, out_path)
    log.info(f"[autodiscover] wrote suggestions → {out_path}")
    return {"path": str(out_path), "sources_found": len(src_top), "loads_found": len(load_top), "device_clues": dev_top}
