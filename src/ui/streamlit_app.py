# -*- coding: utf-8 -*-
import os
import io
import json
import time
import hashlib
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ── path bootstrap (must be first) ─────────────────────────────────────────────
# streamlit_app.py lives at <project>/src/ui/streamlit_app.py
ROOT = Path(__file__).resolve().parents[2]  # -> <project>
if (ROOT / "src").exists() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ───────────────────────────────────────────────────────────────────────────────

# project imports
from src.config.loader import load_cfg
from src.utils.io import read_json, write_json, ensure_dir
from src.utils.logging import setup_logging

# pipeline steps (only run when user clicks)
from src.ingest.pdf_to_svg import extract_svg
from src.ingest.tiler import tile_pages
from src.vision.runners.labels_reader import build_vector_text_index, run_labels_reader
from src.vision.runners.symbol_classifier import classify_meso_tiles
from src.post.merge_candidates import run_merge
from src.geometry.wires import extract_wires_for_pdf
from src.geometry.ports import snap_wires_to_components
from src.stitching.build_nets import stitch_page
from src.refine.violation_detector import detect_violations
from src.refine.recursive_refine import autofix_ports_on_edges

# engineer summaries / exports
from src.summarize.component_summary import component_counts, summarize_components
from src.schema.serialization import export_components_csv

# optional extras (gated in code)
try:
    from src.refine.autodiscover import discover_constraints_candidates
except Exception:
    discover_constraints_candidates = None

try:
    from src.graph.neo4j_adapter import push_page_graph
except Exception:
    push_page_graph = None


# ── UI + helpers ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="diagram-intel", layout="wide", page_icon="🧩")
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
div.metric-value { font-weight: 700 !important; }
</style>
""",
    unsafe_allow_html=True,
)

TITLE = "Diagram Intel — Wiring Diagram Parser"
st.markdown(f"## {TITLE}")

@st.cache_data(show_spinner=False)
def _file_exists(p: str) -> bool:
    return Path(p).exists()

def _stable_run_id_from_name(stem: str) -> str:
    clean = "".join(c if c.isalnum() else "-" for c in stem).strip("-")
    return f"ui-fn-{clean[:40]}"

def _fresh_run_id() -> str:
    return f"ui-{int(time.time())}"

def _save_uploaded_pdf(uploaded, input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    out = input_dir / uploaded.name
    out.write_bytes(uploaded.getvalue())
    return out

def _find_cached_graph_json(cfg, pdf_stem: str):
    # Search in current processed AND in _runs workspaces
    roots = [Path(cfg.paths.processed), Path(cfg.paths.data_root) / "_runs"]
    for root in roots:
        cand = list(root.rglob(f"graphs/{pdf_stem}/page-1.json"))
        if cand:
            return cand[0]
    return None

def _outputs_ready(cfg, pdf_stem: str, page: int) -> bool:
    nets_p = Path(cfg.paths.processed) / "nets" / pdf_stem / f"page-{page}.json"
    graph_p = Path(cfg.paths.processed) / "graphs" / pdf_stem / f"page-{page}.json"
    return nets_p.exists() and graph_p.exists()

def _component_csv_path(cfg, pdf_stem: str, page: int) -> Path:
    return Path(cfg.paths.exports) / pdf_stem / f"components_page-{page}.csv"

def _graphml_path(cfg, pdf_stem: str, page: int) -> Path:
    return Path(cfg.paths.processed) / "graphs" / pdf_stem / f"page-{page}.graphml"

def _graph_json_path(cfg, pdf_stem: str, page: int) -> Path:
    return Path(cfg.paths.processed) / "graphs" / pdf_stem / f"page-{page}.json"

def _plot_net_histogram(nets_json: dict):
    sizes = [n.get("nodes", 0) for n in nets_json.get("nets", [])]
    fig = plt.figure(figsize=(5.8, 3.8))
    plt.hist(sizes, bins=20)
    plt.title("Net size histogram (nodes per net)")
    plt.xlabel("nodes")
    plt.ylabel("count")
    return fig

def _apply_constraints_env(choice: str, pdf_stem: str):
    # Only affects a *new run*
    if choice == "Generic":
        os.environ["CONSTRAINTS_PACKS"] = "generic"
        os.environ.pop("PROJECT_CONSTRAINTS", None)
    elif choice == "Indian power":
        os.environ["CONSTRAINTS_PACKS"] = "generic,indian_power"
        os.environ.pop("PROJECT_CONSTRAINTS", None)
    else:  # Auto-detected
        os.environ["CONSTRAINTS_PACKS"] = "generic,indian_power"
        os.environ["PROJECT_CONSTRAINTS"] = f"suggested_{pdf_stem}"

def _run_pipeline(cfg, pdf_path: Path, pdf_stem: str, constraints_mode: str, do_autofix: bool, reuse: bool, force: bool, page: int = 1):
    # skip compute if user wants reuse and outputs already exist
    if reuse and not force and _outputs_ready(cfg, pdf_stem, page):
        return "cached"

    # 1) ingest + tiles
    extract_svg(str(pdf_path), cfg)
    tile_pages(cfg)

    # 2) labels
    build_vector_text_index(cfg)
    run_labels_reader(cfg)

    # 3) meso classify + merge
    classify_meso_tiles(cfg)
    run_merge(cfg)

    # 4) geometry + nets
    extract_wires_for_pdf(cfg, pdf_stem)
    snap_wires_to_components(cfg, pdf_stem)
    stitch_page(cfg, pdf_stem, page=page)

    # 5) constraints
    if constraints_mode == "Auto-detected" and discover_constraints_candidates is not None:
        try:
            discover_constraints_candidates(cfg)
        except Exception as e:
            st.warning(f"Auto-discover failed (continuing): {e}")
        # reload constraints overlay if created
        cfg = load_cfg()

    try:
        detect_violations(cfg, pdf_stem, page=page)
    except Exception as e:
        st.warning(f"Violation detector skipped: {e}")

    # 6) optional geometry auto-fix
    if do_autofix:
        try:
            autofix_ports_on_edges(cfg, pdf_stem, page=page)
        except Exception as e:
            st.warning(f"Auto-fix skipped: {e}")

    return "ran"

def _render_results(cfg, pdf_stem: str, ui_mode: str, page: int = 1):
    nets_p = Path(cfg.paths.processed) / "nets" / pdf_stem / f"page-{page}.json"
    graph_json_p = Path(cfg.paths.processed) / "graphs" / pdf_stem / f"page-{page}.json"
    graphml_p = Path(cfg.paths.processed) / "graphs" / pdf_stem / f"page-{page}.graphml"
    counts_csv_p = _component_csv_path(cfg, pdf_stem, page)

    if not nets_p.exists() or not graph_json_p.exists():
        st.error("Outputs are missing. Check logs in terminal.")
        return

    nets_json = read_json(nets_p)
    G_json = read_json(graph_json_p)
    n_nets = nets_json.get("count", len(nets_json.get("nets", [])))
    n_nodes = len(G_json.get("nodes", []))
    n_edges = len(G_json.get("edges", []))

    # merged components count (if present)
    merged_idx = Path(cfg.paths.processed) / "components" / "merged" / "merged.index.json"
    n_comps = len(read_json(merged_idx)) if merged_idx.exists() else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nets", n_nets)
    m2.metric("Graph nodes", n_nodes)
    m3.metric("Graph edges", n_edges)
    m4.metric("Merged components", n_comps)

    st.markdown("---")

    # ---------- Engineer mode ----------
    if ui_mode == "Engineer":
        from src.summarize.component_summary import build_device_inventory

        inv_df, details_df = build_device_inventory(cfg, pdf_stem, page=1)

        st.subheader("Device inventory")
        if inv_df.empty:
            st.info("No devices found.")
        else:
            # Clean, compact table: Device | Qty | Typical rating | Example labels
            st.dataframe(inv_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns([0.55, 0.45])

        with c1:
            st.subheader("Per-device details (sample)")
            if not details_df.empty:
                # Show compact columns; users can expand to a full CSV if needed
                show_cols = ["Device","Phase","Ratings","Labels"]
                st.dataframe(details_df[show_cols].head(40), use_container_width=True, hide_index=True)
            else:
                st.caption("(no details)")

        with c2:
            st.subheader("Net size histogram")
            try:
                nets_p = Path(cfg.paths.processed) / "nets" / pdf_stem / "page-1.json"
                nets_json = read_json(nets_p)
                fig = _plot_net_histogram(nets_json)
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.caption(f"(histogram skipped: {e})")

        st.markdown("#### Downloads")
        counts_csv_p = _component_csv_path(cfg, pdf_stem, 1)
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            try:
                # Export a CSV of the inventory table for engineers
                # Reuse components CSV path for simplicity
                inv_out = Path(cfg.paths.exports) / pdf_stem
                inv_out.mkdir(parents=True, exist_ok=True)
                inv_csv = inv_out / f"device_inventory_page-{1}.csv"
                inv_df.to_csv(inv_csv, index=False)
                csv_bytes = inv_csv.read_bytes()
                st.download_button("Download device_inventory.csv",
                                data=csv_bytes,
                                file_name=inv_csv.name,
                                mime="text/csv",
                                disabled=inv_df.empty)
            except Exception as e:
                st.caption(f"(CSV unavailable: {e})")

        with dl2:
            graphml_p = _graphml_path(cfg, pdf_stem, 1)
            gm_bytes = graphml_p.read_bytes() if graphml_p.exists() else b""
            st.download_button("Download page-1.graphml",
                            data=gm_bytes,
                            file_name=graphml_p.name,
                            mime="application/graphml+xml",
                            disabled=not bool(gm_bytes))

        with dl3:
            graph_json_p = _graph_json_path(cfg, pdf_stem, 1)
            gj_bytes = graph_json_p.read_bytes() if graph_json_p.exists() else b""
            st.download_button("Download page-1.graph.json",
                            data=gj_bytes,
                            file_name=graph_json_p.name,
                            mime="application/json",
                            disabled=not bool(gj_bytes))

        st.markdown("#### Integrations")
        if push_page_graph is None:
            st.caption("Neo4j not configured. Add `configs/neo4j.yaml` and `pip install neo4j` to enable a push button here.")
        else:
            if st.button("Push to Neo4j"):
                with st.spinner("Pushing…"):
                    try:
                        res = push_page_graph(cfg, pdf_stem, page=1)
                        st.success(f"Pushed: {res}")
                    except Exception as e:
                        st.error(f"Neo4j push failed: {e}")

    else:
        st.subheader("Nets overview")
        st.json({"count": n_nets, "largest_net_nodes": max((n.get("nodes", 0) for n in nets_json.get("nets", [])), default=0)})
        with st.expander("Show first 5 nets"):
            st.json(nets_json.get("nets", [])[:5])

        st.subheader("Graph snapshot")
        with st.expander("First 5 nodes"):
            st.json(G_json.get("nodes", [])[:5])
        with st.expander("First 5 edges"):
            st.json(G_json.get("edges", [])[:5])

        st.subheader("Files")
        colA, colB, colC = st.columns(3)
        colA.write(_graph_json_path(cfg, pdf_stem, page))
        colB.write(_graphml_path(cfg, pdf_stem, page))
        colC.write(Path(cfg.paths.processed) / "nets" / pdf_stem / f"page-{page}.json")


# ── Sidebar (inputs & guards) ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Input")
    cfg0 = load_cfg()  # base cfg to read paths/input_pdfs
    input_dir = Path(cfg0.paths.input_pdfs)
    uploaded = st.file_uploader("Upload a wiring diagram PDF", type=["pdf"])

    existing = sorted([p.name for p in input_dir.glob("*.pdf")])
    pick_existing = st.selectbox("Or choose an existing PDF", ["—"] + existing, index=0)

    st.markdown("---")
    st.markdown("### Run settings")
    constraints_mode = st.radio("Constraints pack", ["Generic", "Indian power", "Auto-detected"], index=1)
    reuse = st.toggle("Re-use cached results if available", value=True)
    force = st.toggle("Force re-run (ignore cache)", value=False, disabled=reuse)
    do_autofix = st.checkbox("Geometry auto-fix (ports-to-edges)", value=True)

    st.markdown("---")
    ui_mode = st.radio("Display mode", ["Engineer", "Backend"], index=0)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        run_btn = st.button("▶️ Run pipeline", type="primary", disabled=not uploaded and pick_existing == "—")
    with colB:
        load_btn = st.button("📂 Load cached only", disabled=(pick_existing == "—"))

# If no action yet → do nothing
if not run_btn and not load_btn:
    st.info("Upload a PDF or choose an existing one, then click **Run pipeline** or **Load cached only**.")
    st.stop()

# Resolve pdf_path/stem from upload or selection
if uploaded:
    pdf_path = _save_uploaded_pdf(uploaded, Path(cfg0.paths.input_pdfs))
else:
    pdf_path = Path(cfg0.paths.input_pdfs) / pick_existing
pdf_stem = pdf_path.stem

# Load-only branch (no compute)
if load_btn:
    cached_json = _find_cached_graph_json(cfg0, pdf_stem)
    if not cached_json:
        st.warning("No cached results found for this file name. Please run the pipeline.")
        st.stop()
    st.success("Loaded cached results.")
    # Use current cfg for paths (no env change needed)
    _render_results(cfg0, pdf_stem, ui_mode, page=1)
    st.stop()

# Run branch
# Set RUN_ID and constraints env only now
run_id = _stable_run_id_from_name(pdf_stem) if (reuse and not force) else _fresh_run_id()
os.environ["RUN_ID"] = run_id
_apply_constraints_env(constraints_mode, pdf_stem)
cfg = load_cfg()
setup_logging(cfg.logging.level)

# If re-use allowed and cached exists, show it and stop
if reuse and not force:
    if _outputs_ready(cfg, pdf_stem, page=1) or _find_cached_graph_json(cfg, pdf_stem):
        st.success("Using cached results.")
        _render_results(cfg, pdf_stem, ui_mode, page=1)
        st.stop()

with st.spinner("Running pipeline…"):
    status = _run_pipeline(cfg, pdf_path, pdf_stem, constraints_mode, do_autofix, reuse, force, page=1)

st.success("Pipeline finished." if status == "ran" else "Using cached results.")
_render_results(cfg, pdf_stem, ui_mode, page=1)
