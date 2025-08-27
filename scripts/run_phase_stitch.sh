#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)"
python - <<'PY'
from pathlib import Path
from src.config.loader import load_cfg
from src.stitching.build_nets import stitch_page
cfg = load_cfg()
pdf = Path(cfg.paths.input_pdfs)
stem = next(pdf.glob("*.pdf")).stem
stitch_page(cfg, stem, page=1)
print("OK")
PY
