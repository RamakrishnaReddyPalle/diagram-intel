import typer
from src.config.loader import load_cfg
from src.ingest.pdf_to_svg import extract_svg
from src.ingest.tiler import tile_pages
from src.vision.runners.labels_reader import run_labels_reader
from src.stitching.build_nets import stitch
from src.refine.recursive_refine import refine_all
from src.graph.build_graph import build_graph
from src.graph.exporters import export_all

app = typer.Typer()

@app.command()
def run(pdf:str):
    cfg = load_cfg()
    if cfg.phases.ingest: extract_svg(pdf, cfg)
    if cfg.phases.tiling: tile_pages(cfg)
    if cfg.phases.read_labels: run_labels_reader(cfg)
    if cfg.phases.stitch_nets: stitch(cfg)
    if cfg.phases.constraints: refine_all(cfg)
    G = build_graph(cfg)
    if cfg.phases.export_artifacts: export_all(G, cfg)

if __name__ == "__main__":
    app()
