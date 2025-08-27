from pathlib import Path
from typing import List, Dict, Any, Tuple
from svgelements import SVG, Text
from dataclasses import dataclass

BBox = Tuple[float, float, float, float]

@dataclass
class TextItem:
    text: str
    bbox: BBox
    x: float
    y: float

def intersect(b1:BBox, b2:BBox, min_overlap_px:float=1.0) -> bool:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    return (x2 - x1) >= min_overlap_px and (y2 - y1) >= min_overlap_px

def parse_svg_text(svg_path: str, min_chars: int = 2) -> List[Dict[str, Any]]:
    """
    Parse <text>/<tspan> via svgelements with transforms applied.
    Returns list of {text, bbox:[x1,y1,x2,y2], x, y} in SVG coordinate space.
    """
    items: List[Dict[str, Any]] = []
    if not Path(svg_path).exists():
        return items

    svg = SVG.parse(svg_path)
    for el in svg.elements():
        if isinstance(el, Text):
            txt = (el.text or "").strip()
            if not txt or len(txt) < min_chars:
                continue
            bb = el.bbox()
            if bb is None:
                # fallback tiny box around nominal position
                x = float(getattr(el, "x", 0.0))
                y = float(getattr(el, "y", 0.0))
                bbox = (x, y-5.0, x+5.0, y+5.0)
            else:
                bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            items.append({
                "text": txt,
                "bbox": [round(bbox[0],2), round(bbox[1],2), round(bbox[2],2), round(bbox[3],2)],
                "x": round(bbox[0],2),
                "y": round(bbox[1],2),
            })
    return items

# ---------- PyMuPDF fallback ----------
def parse_pdf_text_fitz(pdf_path: str, page_number: int, dpi: int = 900, min_chars: int = 2) -> List[Dict[str, Any]]:
    """
    Use PyMuPDF to read text spans with bboxes on a page (1-based index).
    Coords are converted from points (72 dpi) to our PNG pixel space (cfg.runtime.dpi).
    Returns list of {text, bbox:[x1,y1,x2,y2], x, y} in pixel coords.
    """
    import fitz  # PyMuPDF
    items: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > len(doc):
        return items

    page = doc[page_number - 1]
    scale = dpi / 72.0  # points -> pixels
    # page.get_text("dict") -> blocks -> lines -> spans (with bbox)
    td = page.get_text("dict")
    for block in td.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                if not txt or len(txt) < min_chars:
                    continue
                x0,y0,x1,y1 = span.get("bbox", [0,0,0,0])
                bx = [round(x0*scale,2), round(y0*scale,2), round(x1*scale,2), round(y1*scale,2)]
                items.append({
                    "text": txt,
                    "bbox": bx,
                    "x": bx[0],
                    "y": bx[1],
                })
    return items
