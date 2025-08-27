# src/ingest/tiler.py
from pathlib import Path
from PIL import Image
from src.utils.io import write_json
from src.utils.logging import setup_logging

def _tiles_for_image(img: Image.Image, size: int, overlap: float):
    W, H = img.size
    step = max(1, int(size * (1 - overlap)))
    tiles = []
    y = 0; row = 0
    while y < H:
        x = 0; col = 0
        while x < W:
            x2 = min(x + size, W)
            y2 = min(y + size, H)
            tiles.append((row, col, (x, y, x2, y2)))
            x += step; col += 1
        y += step; row += 1
    return tiles

def tile_pages(cfg):
    log = setup_logging(cfg.logging.level)
    raw_png_root = Path(cfg.paths.raw) / "png"
    out_root = Path(cfg.paths.interim) / "tiles"
    out_root.mkdir(parents=True, exist_ok=True)

    sizes = {
        "micro": cfg.runtime.tile.micro_size,
        "meso":  cfg.runtime.tile.meso_size,
        "macro": cfg.runtime.tile.macro_size,
    }
    overlap = cfg.runtime.tile.overlap

    index = []
    for pdf_folder in sorted(raw_png_root.glob("*")):
        page_pngs = sorted(pdf_folder.glob("page-*.png"))
        if not page_pngs:
            log.warning(f"[tiler] No PNG pages found under {pdf_folder}")
            continue

        for page_png in page_pngs:
            page_id = int(page_png.stem.split("-")[-1])
            img = Image.open(page_png).convert("RGB")
            total_for_page = 0

            for scale, size in sizes.items():
                out_dir = out_root / scale / pdf_folder.name / f"page-{page_id}"
                out_dir.mkdir(parents=True, exist_ok=True)
                tiles = _tiles_for_image(img, size, overlap)

                for (r, c, bbox) in tiles:
                    crop = img.crop(bbox)
                    tpath = out_dir / f"tile_r{r:03d}_c{c:03d}.png"
                    crop.save(tpath)
                    index.append({
                        "pdf": pdf_folder.name,
                        "page": page_id,
                        "scale": scale,
                        "row": r, "col": c,
                        "bbox": bbox,
                        "path": str(tpath)
                    })
                total_for_page += len(tiles)

            log.info(f"[tiler] {pdf_folder.name} page-{page_id}: wrote {total_for_page} tiles")

    write_json(index, out_root / "tile_index.json")
    log.info(f"[tiler] Wrote tile index ({len(index)} rows) → {out_root/'tile_index.json'}")
