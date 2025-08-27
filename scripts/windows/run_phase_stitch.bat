@echo off
set PYTHONPATH=%CD%
python - <<PY
from pathlib import Path
from src.config.loader import load_cfg
from src.stitching.build_nets import stitch_page
cfg = load_cfg()
stem = next(Path(cfg.paths.input_pdfs).glob("*.pdf")).stem
stitch_page(cfg, stem, page=1)
print("OK")
PY
