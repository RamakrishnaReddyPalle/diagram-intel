# **DIAGRAM TO KNOWLEDGE ENGINE**
> Turn raw wiring diagrams into **engineer-friendly** device lists, nets, and graph exports, locally, with open models.
---
## Index

* [1) Problem Statement](#1-problem-statement)
* [2) How i tried to solve it (at a glance)](#2-how-i-tried-to-solve-it-at-a-glance)
* [3) Models: considered vs used](#3-models-considered-vs-used)
* [4) Setup](#4-setup)
* [5) Downloading models (via YAML)](#5-downloading-models-via-yaml)
* [6) Inputs and the “all-stages” notebook](#6-inputs-and-the-all-stages-notebook)
* [7) Controlling which PDF & what parameters](#7-controlling-which-pdf--what-parameters)

  * [Choose the PDF](#choose-the-pdf)
  * [Configuration knobs (quick map)](#configuration-knobs-quick-map)
* [8) Auto-detect candidates (constraints)](#8-auto-detect-candidates-constraints)
* [9) Other controls (graphs, Neo4j, etc.)](#9-other-controls-graphs-neo4j-etc)
* [10) Running the app](#10-running-the-app)
* [11) What the app gives you](#11-what-the-app-gives-you)
* [12) Constraints: what i checked](#12-constraints-what-i-checked)
* [13) Future scope](#13-future-scope)
* [Quickstart (zero to results)](#quickstart-zero-to-results)
* [Where files end up](#where-files-end-up)
---
## **1) Problem Statement**

Electrical drawings are packed with meaning, but it’s locked up in a mix of vector lines, scattered text, repeated legends, and sometimes scanned content. Engineers want simple answers:

* *What devices are on this sheet?*
* *How many of each?*
* *Which nets exist and how big are they?*
* *Can I export a graph / netlist into downstream tools?*

Manual extraction is slow and error-prone. That’s the gap **Diagram-Intel** closes.

---
<p align="center">
  <img src="assets/Screenshot 2025-08-27 205303.png" style="width: 100%;">
</p>

## **2) How i tried to solve it (at a glance)**

| Stage         | What it does                                                          | Output                                   |                 |
| ------------- | --------------------------------------------------------------------- | ---------------------------------------- | --------------- |
| Ingest        | Render PDF pages → SVG/PNG at controlled DPI                          | `data/.../raw/...` manifests             |                 |
| Tiling        | Create micro/meso tiles for local reasoning                           | `interim/tiles/` + index                 |                 |
| Label read    | **Vector text** first (from SVG/PDF spans), with **OCR/VLM fallback** | `processed/labels/tiles/*.json`          |                 |
| Symbol typing | Classify meso tiles as device candidates                              | `processed/components/candidates/*.json` |                 |
| Merge         | De-duplicate overlapping candidates into components                   | `processed/components/merged/*.json`     |                 |
| Geometry      | Detect wires/junctions, snap ports to components                      | \`processed/wires                        | ports/\*.json\` |
| Stitch        | Build nets; export a page graph + GraphML                             | \`processed/nets                         | graphs/...\`    |
| Refine        | Rule-based checks (phase, neutral, RCCB isolation, etc.)              | `processed/refine/...violations.json`    |                 |
| Summaries     | **Engineer-readable** inventory & CSV                                 | `exports/<pdf>/components_page-#.csv`    |                 |
| App           | Streamlit UI with **Engineer** and **Backend** modes                  | Runs locally, reuses cache               |                 |

Key design choices:

* **Local-first** (HF models cached to `models/`).
* **Vector-then-model**: prefer SVG text; only ask a VLM when needed.
* **Constraints packs**: portable defaults + region-specific overlays.
* **Cache by file name** for instant re-open; **RUN\_ID** workspaces for experiments.
---

## **3) Models: considered vs used**

| Area                  | Considered (available in repo)                                    | Why interesting                                                           | Used in this build                                                                                                      |
| --------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Vision-Language (VLM) | Qwen2-VL-2B / 7B, LLaVA-1.6-Mistral-7B, Florence-2, MiniCPM-V 2.6 | Good at short JSON answers; some allow strict prompting for tables/labels | **Qwen2-VL-2B** (primary, fast CPU), **Qwen2-VL-7B** (optional heavier); **LLaVA-1.6-Mistral-7B** for quick smoke tests |
| OCR/fallback          | Donut, Nougat, Pix2Struct                                         | Robust with scanned content and table-ish text                            | **Donut-base** used as OCR fallback when vector text is thin                                                            |
| CV wiring             | (planned) Hough/skeletonization/merge                             | Classical CV to trace polylines, junctions                                | Custom code in `src/geometry` (already used)                                                                            |
| Graph/Query           | Neo4j (optional), NetworkX                                        | Graph queries and export                                                  | NetworkX + GraphML export (Neo4j optional switch)                                                                       |

> Default stack (4 models actually used): **Qwen2-VL-2B**, **Qwen2-VL-7B** (optional), **LLaVA-1.6-Mistral-7B** (optional smoke test), **Donut-base** (OCR fallback).

---

## **4) Setup**

| Step                       | Command / Notes                                 |
| -------------------------- | ----------------------------------------------- |
| Create venv                | `pip install -r requirements.txt`               |
| (Windows) VC++ build tools | If needed for some deps (e.g. PyMuPDF)          |
| Optional GraphML speedup   | `pip install lxml`                              |
| Optional Neo4j client      | `pip install neo4j`                             |
| (Jupyter widgets for tqdm) | `pip install ipywidgets` then enable in Jupyter |

**Project layout** (high-level):

```
diagram-intel/
  configs/           # yaml configs (paths, models, pipeline, constraints, streamlit)
  data/              # input_pdfs/, raw/, interim/, processed/, exports/
  models/            # cache/ + registry.json
  notebooks/         # dev & smoke tests
  scripts/           # download_models.py, run_* (optional)
  src/               # the code
```

---

## 5) **Downloading models (via YAML)**

**Where:** `configs/models.yaml` controls which repos to fetch.
**How:** use the helper script; it reads that YAML and pulls from Hugging Face into `models/cache/`, then records **local paths** in `models/registry.json`.

```bash
# Example
python scripts/download_models.py
# Windows
.\scripts\windows\download_models.bat
```

**Verify:**

* `models/registry.json` will list each model and its `local_path`.
* Your `configs/models.yaml` entries will be “resolved” into the registry.

**Approx disk footprint** (varies by quantization / format; check HF for exact):

| Model                | Rough size on disk | Path after download                      |
| -------------------- | ------------------ | ---------------------------------------- |
| Qwen2-VL-2B-Instruct | \~3–6 GB           | `models/cache/Qwen2-VL-2B-Instruct/`     |
| Qwen2-VL-7B-Instruct | \~8–16 GB          | `models/cache/Qwen2-VL-7B-Instruct/`     |
| LLaVA-1.6-Mistral-7B | \~8–16 GB          | `models/cache/llava-v1.6-mistral-7b-hf/` |
| Donut-base           | \~0.5–1 GB         | `models/cache/donut-base/`               |

*(Numbers are ballpark — formats/weights change. The script prints the exact folder.)*

---

## **6) Inputs and the “all-stages” notebook**

1. Put your PDFs into:

```
data/input_pdfs/
```

2. Open the **smoke-test** notebook (the dev harness you used). It runs all stages into an isolated workspace using `RUN_ID` so your previous results stay intact.

* **Isolation:** we store outputs under:

```
data/_runs/<RUN_ID>/{raw,interim,processed,exports}
```

* **Where to set `RUN_ID`:** in the first cell of the dev notebook or via environment:

```python
import os
os.environ["RUN_ID"] = "dev-session-001"
```

3. The harness drives: ingest → tiles → labels (vector+OCR) → symbols → merge → geometry → nets → refine → summaries → exports.

---

## **7) Controlling which PDF & what parameters**

### **Choose the PDF**

* The harness/notebook picks the first file in `data/input_pdfs/` by default — or you can set it explicitly by path (or pick in the Streamlit app).

### **Configuration knobs (quick map)**

| File                                  | Key                                                                                    | What it affects                               |
| ------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------- |
| `configs/paths.yaml`                  | data/model roots                                                                       | Folder locations (input/output)               |
| `configs/base.yaml`                   | device, precision, logging                                                             | CPU/GPU choice, float32/16, log level         |
| `configs/pipeline.yaml`               | DPI, tile sizes, thresholds                                                            | Render DPI, micro/meso tile grid, VLM budgets |
| `configs/models.yaml`                 | HF repos, local overrides                                                              | Which models to use and where                 |
| `configs/constraints/packs/*.yaml`    | `generic`, `indian_power`                                                              | Electrical rule/heuristic overlays            |
| `configs/constraints/projects/*.yaml` | `suggested_<PDF>.yaml`                                                                 | Auto-discovered project constraints           |
| `.env`                                | `RUN_ID`, `CONSTRAINTS_PACKS`, `PROJECT_CONSTRAINTS`, `DEVICE`, `PRECISION`, `HF_HOME` | Runtime switches without editing YAML         |

**Typical examples:**

```bash
# Use generic+Indian constraints; apply auto-detected overlay for this PDF:
CONSTRAINTS_PACKS=generic,indian_power
PROJECT_CONSTRAINTS=suggested_<Your PDF Stem>

# Force CPU + float32 for max compatibility:
DEVICE=cpu
PRECISION=float32
```

---

## 8) **Auto-detect candidates (constraints)**

You can have the system scan labels and **suggest** a project overlay:

* In code: `discover_constraints_candidates(cfg)` writes
  `configs/constraints/projects/suggested_<PDF_STEM>.yaml`.

* To **use** it across runs, set:

```
CONSTRAINTS_PACKS=generic,indian_power
PROJECT_CONSTRAINTS=suggested_<PDF_STEM>
```

* In the app, choose **“Auto-detected”** from the **Constraints pack** toggle.

---

## **9) Other controls (graphs, Neo4j, etc.)**

| Feature               | How to enable / control                                                                                   | Output                                  |
| --------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| GraphML export        | (Default on) NetworkX writes `.graphml` alongside JSON                                                    | `processed/graphs/<pdf>/page-1.graphml` |
| Faster GraphML        | `pip install lxml`                                                                                        | Faster `nx.write_graphml`               |
| JSON graph            | Always written                                                                                            | `processed/graphs/<pdf>/page-1.json`    |
| Neo4j push (optional) | Fill `configs/neo4j.yaml`, `pip install neo4j` — app shows a “Push to Neo4j” button if adapter is present | Your DB                                 |
| OCR fallback          | Handled inside label reader when vector text is sparse                                                    | Tile label JSON                         |
| VLM budgets           | `pipeline.yaml` (e.g., `labels.max_tiles_vlm`)                                                            | Speed vs recall                         |

---

## **10) Running the app**

```bash
streamlit run src/ui/streamlit_app.py
```

If Streamlit cannot find your package, the app already bootstraps the project root; no extra `PYTHONPATH` needed.

---

## **11) What the app gives you**

**Two modes**

| Mode         | What you see                                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Engineer** | **Device inventory** (Device, Qty, Typical rating, Example labels), sample **per-device details**, **net size histogram**, and one-click **CSV/GraphML/JSON** downloads. |
| **Backend**  | Raw nets and graph snapshots (first N nodes/edges), file paths for debugging.                                                                                            |

**Input & caching workflow**

| Control                     | Behavior                                            |
| --------------------------- | --------------------------------------------------- |
| Upload or pick existing PDF | Use `data/input_pdfs/` or upload directly           |
| **Re-use cached results**   | Uses prior outputs keyed by file name (fast)        |
| **Force re-run**            | Recomputes everything in a fresh `RUN_ID` workspace |
| **Constraints pack**        | **Generic**, **Indian power**, or **Auto-detected** |
| **Geometry auto-fix**       | Snap “floating” ports to component edges            |

**Downloads & integrations**

* `device_inventory.csv` (Engineer-readable)
* `page-1.graphml` and `page-1.graph.json`
* Optional **Push to Neo4j** (if configured)

---

## **12) Constraints: what i checked**

The rulebook is split into **portable** and **regional** overlays:

| Pack           | Examples inside                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `generic`      | Device port expectations (e.g., MCCB 3P/4P, RCCB 2P/4P), neutral presence policies, phase consistency, junction rules                           |
| `indian_power` | LV three-phase conventions (L1/L2/L3/N), common device names (ACCL, TPN), vocabulary and color hints drawn from widely used Indian/IEC practice |

What i flag today (examples):

* **Giant nets** (warn/error thresholds)
* **RCCB not isolating** (all neighbors on 1 net → likely bypass)
* **Source bridges without changeover** (e.g., “EB” & “DG” & “PV” signatures on one net but no ACCL/ATS/3-way selector present)

> All of this is **editable** in `configs/constraints/packs/*.yaml` and project overlays. Treat it as a living rulebook you can tailor to your organization and standards library.

---

## **13) Future scope**

| Track             | Ideas                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------- |
| Better typing     | Train a small local detector/segmenter for symbol regions; fuse with VLM for high-precision device types |
| Scanned drawings  | Stronger OCR ensemble, denoising, skew/deskew + CV snap-to-grid                                          |
| Multi-sheet links | Cross-references across pages; device instance linking; hierarchy                                        |
| Rich exports      | SPICE-like netlists, JSON-LD vocab for devices, queries over Neo4j                                       |
| Constraints       | Deeper inference (Z3 solvers), rating propagation, protection selectivity checks                         |
| UI                | Multi-RUN compare, inline tile/gallery debugger, manual fix-ups and re-stitch                            |

---

## **Quickstart (zero to results)**

```bash
# 1) Install
pip install -r requirements.txt

# 2) Download models from configs/models.yaml
python scripts/download_models.py

# 3) Put a PDF in:
#    data/input_pdfs/YourDiagram.pdf

# 4) (Option A) Run the Streamlit app
streamlit run src/ui/streamlit_app.py

# 4) (Option B) Use the dev harness (isolated run)
# In the first cell:
#   os.environ["RUN_ID"] = "my-test-001"
#   os.environ["CONSTRAINTS_PACKS"] = "generic,indian_power"
#   os.environ["PROJECT_CONSTRAINTS"] = "suggested_<Your PDF Stem>"  # optional
# Then step through cells to produce processed outputs and exports.
```

---

### **Where files end up**

| Kind                   | Path                                                  |        |
| ---------------------- | ----------------------------------------------------- | ------ |
| Input PDFs             | `data/input_pdfs/`                                    |        |
| One-shot runs          | `data/processed/...`                                  |        |
| Isolated dev runs      | `data/_runs/<RUN_ID>/{raw,interim,processed,exports}` |        |
| Graph exports          | \`processed/graphs/<pdf>/page-1.graphml               | json\` |
| Engineer CSV           | `exports/<pdf>/components_page-#.csv`                 |        |
| Device inventory (app) | `exports/<pdf>/device_inventory_page-#.csv`           |        |
| Model cache            | `models/cache/` (see `models/registry.json`)          |        |
