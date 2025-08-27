import argparse
from pathlib import Path
from src.config.loader import load_cfg
from src.stitching.build_nets import stitch_page

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=str, default=None, help="PDF stem (without .pdf). If not set, takes first input.")
    ap.add_argument("--page", type=int, default=1)
    args = ap.parse_args()

    cfg = load_cfg()
    if args.pdf is None:
        first = next(Path(cfg.paths.input_pdfs).glob("*.pdf"), None)
        assert first is not None, "No PDFs in input_pdfs."
        pdf_stem = first.stem
    else:
        pdf_stem = args.pdf

    stitch_page(cfg, pdf_stem, page=args.page)

if __name__ == "__main__":
    main()
