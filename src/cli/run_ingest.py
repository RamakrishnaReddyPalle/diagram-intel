import typer
from pathlib import Path
from src.config.loader import load_cfg
from src.ingest.pdf_to_svg import extract_svg
from src.ingest.tiler import tile_pages

app = typer.Typer()

@app.command()
def run(pdf: str):
    cfg = load_cfg()
    pdf_path = Path(pdf)
    if not pdf_path.is_absolute():
        pdf_path = (Path(cfg.root) / pdf_path).resolve()
    print(f"[ingest] Using PDF: {pdf_path}")
    extract_svg(str(pdf_path), cfg)
    tile_pages(cfg)

if __name__ == "__main__":
    app()
