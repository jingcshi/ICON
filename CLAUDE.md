# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

ICON (Implicit CONcept Insertion) is a self-supervised taxonomy enrichment system. It inserts implicit intermediate concepts into an existing taxonomy by:
1. Retrieving a cluster of related concepts using an embedding model (`emb_model`)
2. Generating a "virtual concept" label representing their semantic union via a generative model (`gen_model`)
3. Placing the new concept via subsumption-prediction-based search using a subsumption model (`sub_model`)

## Code architecture

```
src/icon/
  __init__.py          — Public API: ICON, ICONforCategoryMove, Taxonomy, TreeTaxonomy, from_json, from_owl
  core/
    taxonomy.py        — Taxonomy(nx.DiGraph): core graph class; TreeTaxonomy subclass; from_json(), from_owl()
    icon.py            — ICON class: run(), auto(), semiauto(), manual(), outer_loop(), inner_loop(),
                          enhanced_traversal(), insert()
    category_move.py   — ICONforCategoryMove(ICON): specialised subclass for concept re-ranking
  config/
    config.py          — All config dataclasses (tree_config hierarchy); update_config() for dynamic updates
  models/              — Sub-model wrappers / ABC for emb_model, gen_model, sub_model
  utils/
    vector_index.py    — FaissVectorStore: wraps FAISS IVF index for concept embedding search
    tokenset_utils.py  — NLP utilities: tokenset(), lemmatization, breadcrumb normalisation
    log_style.py       — ANSI color constants for console logging
  cli/
    main.py            — Click command group; `icon taxo view/convert/validate`

gui/
  app.py               — Entry point (strips /opt/clients from sys.path, runs Dash server)
  dash_app.py          — App factory: create_app() with serve_locally=True, no CDN stylesheets
  layout.py            — build_layout() returns the full Dash component tree
  callbacks.py         — All Dash callbacks; register(app) is the entry point
  assets/
    bootstrap.min.css  — Bootstrap served locally (no CDN); required because the VM has no internet

tests/                 — pytest unit tests for Taxonomy graph ops and FaissVectorStore
experiments/           — Notebooks and scripts for data wrangling, model training, evaluation
```

## Common commands

**Run the GUI:**
```bash
python gui/app.py
# Access at http://<VM-IP>:8050 (not localhost — see proxy note below)
```

**Run CLI:**
```bash
icon taxo view <file.json>       # stats + tree preview
icon taxo convert <src> <dst>    # JSON ↔ OWL
icon taxo validate <file>        # DAG integrity check
```

**Run tests:**
```bash
pytest tests/                    # all tests
pytest tests/test_taxonomy.py    # single module
```

**Lint / type-check:**
```bash
ruff check src/ tests/
mypy src/
```

## Key design patterns

**Config system:** All ICON parameters are nested frozen dataclasses (`tree_config` hierarchy in `config/config.py`). `update_config(**kwargs)` resolves kwargs via `locate_arg()` and returns a new config via `recursive_replace()`. Field names must be unique across the tree.

**Taxonomy graph:** `Taxonomy` subclasses `nx.DiGraph`. Edges `(u, v)` represent `u subClassOf v` (child→parent direction). Node key `0` is always the root. `get_parents(node)` uses `self._succ`; `get_children(node)` uses `self._pred`. `add_edge` raises `NetworkXError` on cycles.

**Edge direction in subgraphs:** In any NetworkX subgraph, `successors(node)` = **parents** (upstream), `predecessors(node)` = **children** (downstream). This is the opposite of the name's intuition.

**Three operating modes:**
- `auto` — iterates all bottom-level concepts as seeds; requires all three models
- `semiauto` — user-specified seed list; requires all three models
- `manual` — places user-specified labels directly; requires only `sub_model`

**GUI proxy constraint:** The VM has intranet-only access. All CDN fetches (jsdelivr, unpkg, dash-version.plotly.com) block indefinitely. The GUI uses `serve_locally=True`, `external_stylesheets=[]`, and `dev_tools_disable_version_check=True` to avoid all CDN requests. Bootstrap is downloaded to `gui/assets/bootstrap.min.css` and served locally by Dash.

**GUI access:** The Krylov proxy URL and VS Code Simple Browser both fail (connection refused / proxy authentication). Access via direct intranet IP: `http://<VM-IP>:8050` from a device on the same network.

**`/opt/clients` sys.path pollution:** The Krylov VM injects `/opt/clients/pykrylov/…` into PYTHONPATH at startup, which ships a broken pydantic. `gui/app.py` strips these entries before any import.

## Taxonomy JSON format

```json
{
  "nodes": [{"id": 1, "label": "Electronics"}, ...],
  "edges": [{"src": 42, "tgt": 1, "label": "original"}, ...]
}
```
- `src` = child node ID, `tgt` = parent node ID
- Node `id=0` is reserved for root (auto-created if absent)
- Extra fields on nodes/edges are preserved as attributes

## Dependencies

**Core:** `numpy`, `networkx`, `rdflib`, `faiss-gpu`, `nltk`, `tqdm`, `click`, `pyyaml`  
**GUI:** `dash`, `dash-cytoscape`, `dash-bootstrap-components`  
**Training:** `torch`, `transformers`, `datasets`, `evaluate`, `info-nce-pytorch`, `pandas`, `scikit-learn`  
**Dev:** `pytest`, `ruff`, `mypy`  

See `pyproject.toml` for pinned optional dep groups (`[gui]`, `[training]`, `[dev]`).  
See `environment-full.yml` / `environment-gui.yml` for reproducible conda environments.

Python ≥ 3.9 required.
