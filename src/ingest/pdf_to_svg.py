# src/ingest/pdf_to_svg.py
from pathlib import Path
from src.utils.logging import setup_logging
from src.utils.io import write_json
from src.utils import pdf as pdfu

def extract_svg(pdf: str, cfg):
    log = setup_logging(cfg.logging.level)
    pdf_path = Path(pdf)
    raw_dir = Path(cfg.paths.raw)
    svg_dir = raw_dir / "svg" / pdf_path.stem
    png_dir = raw_dir / "png" / pdf_path.stem
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[ingest] Exporting SVG/PNG for {pdf_path.name} (dpi={cfg.runtime.dpi})")
    manifest = pdfu.export_svg_and_png(str(pdf_path), str(svg_dir), str(png_dir), dpi=cfg.runtime.dpi)

    # Add high-level summary fields
    n_pages = manifest.get("num_pages", 0)
    n_vectorish = sum(1 for p in manifest["pages"] if p.get("vector_like"))
    manifest.update({
        "svg_dir": str(svg_dir),
        "png_dir": str(png_dir),
        "vectorish_pages": n_vectorish,
    })

    out_manifest = raw_dir / "manifests" / f"{pdf_path.stem}.json"
    write_json(manifest, out_manifest)
    log.info(f"[ingest] Engine={manifest['engine']} | pages={n_pages} | vector-like pages={n_vectorish}")
    log.info(f"[ingest] Wrote manifest → {out_manifest}")
