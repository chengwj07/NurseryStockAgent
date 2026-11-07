# _From UAV Imagery to Nursery Stock Value with Vision Foundation Models and an LLM Agent_

The **Nursery Stock Agent** is a research prototype for extracting, inventorying, and valuing nursery stock from UAV imagery through a natural-language interface. It combines a **Gradio** interface, a **Leaflet** map, a **LangChain-based Planner–Executor agent**, **SAM3**-based crown extraction, and deterministic Python tools for counting, valuation, and report generation.

## 1. Features

| Tool module | Registered tool | Function |
| --- | --- | --- |
| `tools/extract_tool.py` | `extract_trees` | Extract crowns for a specified nursery-stock type using SAM3 and display the results as a map layer. |
| `tools/count_tool.py` | `count_trees` | Count extracted plants overall or by species, optionally within a selected area. |
| `tools/valuation_tool.py` | `evaluate_value` | Estimate per-plant and aggregate values using species-specific equations. |
| `tools/table_tool.py` | `generate_table` | Export Excel workbooks using the `asset_ledger` or `plot_census` template. |
| `tools/report_tool.py` | `generate_report` | Generate a Word inventory report for a selected area. |

## 2. System architecture

```text
Natural-language request
          |
          v
   Planner (LLM)
   - interpret intent
   - select preset or compose tool steps
   - return structured plan (JSON) or direct reply
          |
          v
   Executor (no LLM)
          |
          v
   LangChain StructuredTools --> Python tools (`tools/*.py`)
          |
          v
   SessionState (images, masks/polygons, layers, valuations)
          |
          +----------------------> Leaflet map / downloadable files

   On execution failure: error feedback --> Planner --> revised plan
   (up to the configured `max_replans` limit)
```

## 3. Repository structure

```text
app.py                  # Entry point: FastAPI application with mounted Gradio UI
config.py               # Settings from environment variables / .env (NURSERY_ prefix)
logging_config.py       # Logging configuration
agent/
  llm_factory.py        # OpenAI-compatible API or local llama.cpp model
  planner.py            # Prompt -> LLM -> parser -> structured plan
  presets.py            # Predefined workflows
  registry.py           # Automatic tool discovery and registration
  executor.py           # Deterministic plan execution
  service.py            # Planning, execution, replanning, streamed events
  state.py              # SessionState
tools/                  # One tool per *_tool.py module
services/               # SAM3 segmentation, pricing, storage, tiling, statistics
ui/                     # Gradio layout, Leaflet map, tile/GeoJSON endpoints
templates/
  price_table.json      # Species aliases, SAM3 prompts, valuation formulas
  table_templates.json  # Excel report templates
tests/                  # Smoke tests with a fake LLM and mock segmentation
```

## 4. Installation and quick start

### Requirements

- Python environment with dependencies listed in `requirements.txt`.
- SAM3 installed according to its [official repository](https://github.com/facebookresearch/sam3) for real segmentation.
- An accessible LLM api key, or a local GGUF model with `llama-cpp-python`.
- GPU resources as appropriate for the chosen segmentation and LLM backends.

```bash
pip install -r requirements.txt
# Install SAM3 separately following its official instructions.
# For the local llama.cpp mode only:
pip install llama-cpp-python

cp .env.example .env
# Edit .env for your environment and model backends.
python app.py
```

Open **http://127.0.0.1:7860** in a browser.

For interface development without a segmentation model, set `NURSERY_SEG_BACKEND=mock`. **Mock mode generates synthetic/random crowns and must not be used to report scientific results.** An LLM configuration is still required.

### LLM configuration

**Option A: OpenAI-compatible API**

In the project-root `.env` file:

```dotenv
NURSERY_LLM_MODE=api
NURSERY_API_BASE_URL=https://YOUR_PROVIDER_ENDPOINT/v1
NURSERY_API_KEY=YOUR_API_KEY
NURSERY_API_MODEL=YOUR_MODEL_NAME
```

The API endpoint can also point to a local `llama-server` exposing an OpenAI-compatible `/v1` interface.

**Option B: Local llama.cpp**

```dotenv
NURSERY_LLM_MODE=llamacpp
NURSERY_LLAMACPP_MODEL_PATH=/path/to/model.gguf
```

Configure context size and GPU offloading through the corresponding `NURSERY_` settings in `.env.example` (including `N_CTX` and `N_GPU_LAYERS` options). Model identifiers and paths are configuration-driven rather than hard-coded. The model construction logic resides in `agent/llm_factory.py`.

## 5. Typical workflow

1. **Upload imagery.** Load a georeferenced UAV orthomosaic (GeoTIFF) or a supported ordinary image. Optionally associate an DEM image with the selected image.
2. **Request extraction.** Ask the agent to extract one of the supported nursery-stock types. SAM3 predictions are postprocessed into mapped crown polygons.
3. **Count plants.** Request counts for the entire image, by species, or within a selected region.
4. **Estimate asset value.** Run the valuation tool to compute per-plant prices and species/area totals where valid formulas and required measurements are available.
5. **Export results.** Generate an Excel asset ledger or plot census, or a Word report.

Example requests:

```text
Extract podocarpus macrophyllus from the selected image and count them.
Estimate the total value of the extracted nursery stock.
Generate an asset ledger for the selected area.
Create a nursery-stock inventory report.
```

Parameters can be adjusted in `.env`:

| Variable | Default | Description |
| --- | ---: | --- |
| `NURSERY_MAX_HOLE_AREA_M2` | 1 m^2 | Max hole filling threshold. |
| `NURSERY_MIN_OBJECT_AREA_M2` | 1 m^2 | Min debris cleaning-up threshold. |
| `NURSERY_TILE_OVERLAP` | 128 | Sliding window overlap pixels|
| `NURSERY_PMF_CELL_SIZE_M` | 1 m | Grid cell size for the minimum-value reduction before filtering. |
| `NURSERY_PMF_MAX_WINDOW_M` | 20 m | Maximum morphological filter window. |
| `NURSERY_PMF_SLOPE` | 0.15 | Allowed terrain slope parameter. |
| `NURSERY_PMF_INITIAL_DH_M` | 0.15 m | Initial height-difference threshold. |
| `NURSERY_PMF_MAX_DH_M` | 3 m | Maximum height-difference threshold. |

## 6. Extending the agent

**Add a tool.** Create a `*_tool.py` file under `tools/`, subclass `BaseTool`, and implement its metadata, argument schema, and execution method. Restart the application to register the new tool automatically.

```python
class MyArgs(RegionArgs):
    foo: str = Field(..., description="...")

class MyTool(BaseTool):
    name = "my_tool"
    description = "..."
    args_schema = MyArgs

    def run(self, ctx: ToolContext, args: MyArgs) -> ToolResult:
        ...
```

**Add a preset workflow.** Register a `PresetPlan` using `PresetRegistry.register(PresetPlan(name, description, params, builder))`.

**Add an Excel template.** Define its title, metadata, columns, and totals in `templates/table_templates.json` without changing application code.

**Change species or valuation equations.** Edit `templates/price_table.json`. Each species can define SAM3 `prompt` text, `aliases`, and optional `formulas`; formula coefficients can use `coef` or `ln_coef` (interpreted as `exp(ln_coef)`).

## 7. Citation

If this demo supports your research, please cite the associated manuscript:

> *From UAV Imagery to Nursery Stock Value with Vision Foundation Models and an LLM Agent.*

**Bibliographic details and BibTeX:** to be added after the manuscript's author list, venue, year, and DOI are finalized.

## 8. Acknowledgments and third-party software

This application builds on [SAM3](https://github.com/facebookresearch/sam3), [LangChain](https://github.com/langchain-ai/langchain), [Gradio](https://github.com/gradio-app/gradio), [FastAPI](https://github.com/fastapi/fastapi), and [Leaflet](https://leafletjs.com/), among other dependencies. Please consult their respective licenses and model terms. A license for **this repository** should be specified separately before public release.
