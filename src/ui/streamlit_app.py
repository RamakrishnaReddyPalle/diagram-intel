# simple end-to-end demo
import streamlit as st
from pathlib import Path
from src.config.loader import load_cfg
from src.cli.run_full import run as run_pipeline
from src.graph.queries import paths_to_load, list_devices_on_path
from src.utils.io import load_json

st.set_page_config(page_title="Diagram Intel", layout="wide")
st.title("Diagram Intelligence – Electrical Drawing to Graph")

cfg = load_cfg()
uploaded = st.file_uploader("Upload wiring PDF", type=["pdf"])
if uploaded:
    pdf_path = Path("data/input_pdfs")/uploaded.name
    pdf_path.write_bytes(uploaded.getbuffer())
    with st.spinner("Running pipeline..."):
        run_pipeline(str(pdf_path))

    # show outputs
    st.subheader("Component summaries")
    # assume we dump components JSON here:
    # render tables
    # ...
    st.subheader("Trace example: Feeder to Lift")
    src_node = st.text_input("Source node id", "")
    dst_node = st.text_input("Lift node id", "")
    if st.button("Trace path") and src_node and dst_node:
        path = paths_to_load(src_node, dst_node, cfg)
        st.write(path)
        st.write(list_devices_on_path(path, cfg))
