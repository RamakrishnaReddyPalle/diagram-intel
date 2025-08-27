# pdf triage, pdftocairo wrappers
# src/utils/pdf.py
from pathlib import Path
import subprocess, shutil
from typing import List, Dict, Any, Tuple, Optional

def has_pdftocairo() -> bool:
    # On Windows + conda, pdftocairo lives under <env>\Library\bin
    return shutil.which("pdftocairo") is not None

def _call_pdftocairo_svg_png(pdf_path: str, out_svg_dir: str, out_png_dir: str, dpi: int) -> Dict[str, Any]:
    """Export per-page SVG and PNG via pdftocairo. Returns manifest-like info."""
    Path(out_svg_dir).mkdir(parents=True, exist_ok=True)
    Path(out_png_dir).mkdir(parents=True, exist_ok=True)

    # SVG
    # Writes page-1.svg, page-2.svg, ...
    svg_cmd = ["pdftocairo", "-svg", pdf_path, str(Path(out_svg_dir) / "page")]
    # PNG
    # Writes page-1.png, ...
    png_cmd = ["pdftocairo", "-png", "-r", str(dpi), pdf_path, str(Path(out_png_dir) / "page")]

    subprocess.check_call(svg_cmd)
    subprocess.check_call(png_cmd)

    # pdftocairo doesn't return sizes; we gather later in caller
    return {"engine": "pdftocairo"}

def _export_with_pymupdf(pdf_path: str, out_svg_dir: str, out_png_dir: str, dpi: int) -> Dict[str, Any]:
    """Fallback using PyMuPDF: render SVG + PNG per page."""
    import fitz  # PyMuPDF

    Path(out_svg_dir).mkdir(parents=True, exist_ok=True)
    Path(out_png_dir).mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc, start=1):
        # PNG raster
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_path = Path(out_png_dir) / f"page-{i}.png"
        pix.save(png_path.as_posix())

        # SVG vector-ish (PyMuPDF’s SVG writer). May rasterize some content but preserves vectors often.
        svg_str = page.get_svg_image(matrix=fitz.Matrix(1, 1))  # independent of dpi
        svg_path = Path(out_svg_dir) / f"page-{i}.svg"
        svg_path.write_text(svg_str, encoding="utf-8")

    return {"engine": "pymupdf", "pages": len(doc)}

def export_svg_and_png(pdf_path: str, out_svg_dir: str, out_png_dir: str, dpi: int = 900) -> Dict[str, Any]:
    """
    Export each page to SVG and high-DPI PNG.
    Tries pdftocairo first; falls back to PyMuPDF. Returns a manifest dict with per-page info.
    """
    pdf_path = str(pdf_path)
    out_svg_dir = str(out_svg_dir)
    out_png_dir = str(out_png_dir)

    manifest: Dict[str, Any] = {"pdf": pdf_path, "dpi": dpi, "pages": []}

    # Try pdftocairo; fallback to PyMuPDF if missing or fails
    used_engine = None
    if has_pdftocairo():
        try:
            _call_pdftocairo_svg_png(pdf_path, out_svg_dir, out_png_dir, dpi)
            used_engine = "pdftocairo"
        except Exception as e:
            used_engine = f"pdftocairo_failed:{e.__class__.__name__}"
    if used_engine is None or used_engine.startswith("pdftocairo_failed"):
        info = _export_with_pymupdf(pdf_path, out_svg_dir, out_png_dir, dpi)
        used_engine = info.get("engine", "pymupdf")

    # Build per-page metadata
    # We’ll detect "vector-ish" pages by checking SVG file size and whether it contains at least one <path> or <text>.
    svg_root = Path(out_svg_dir)
    png_root = Path(out_png_dir)

    page_idx = 1
    while True:
        svg_p = svg_root / f"page-{page_idx}.svg"
        png_p = png_root / f"page-{page_idx}.png"
        if not png_p.exists():  # stop when PNG missing; both should be in sync
            break

        # Basic vector heuristic
        vector = False
        if svg_p.exists():
            try:
                s = svg_p.read_text(encoding="utf-8", errors="ignore")
                if ("<path" in s) or ("<text" in s) or ("<line" in s) or ("<polyline" in s):
                    vector = True
            except Exception:
                pass

        # Get image size (lazy: PIL read)
        try:
            from PIL import Image
            W, H = Image.open(png_p).size
        except Exception:
            W, H = None, None

        manifest["pages"].append({
            "page": page_idx,
            "svg": svg_p.as_posix() if svg_p.exists() else None,
            "png": png_p.as_posix(),
            "vector_like": bool(vector),
            "size": [W, H],
        })
        page_idx += 1

    manifest["engine"] = used_engine
    manifest["num_pages"] = len(manifest["pages"])
    return manifest
