# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

ICON (Implicit CONcept Insertion) is a self-supervised taxonomy enrichment system. It inserts implicit intermediate concepts into an existing taxonomy by:
1. Retrieving a cluster of related concepts using an embedding model (`emb_model`)
2. Generating a "virtual concept" label representing their semantic union via a generative model (`gen_model`)
3. Placing the new concept via subsumption-prediction-based search using a subsumption model (`sub_model`)

The main entry point is `main/icon.py::ICON`. A derived class `main/category_move.py::ICONforCategoryMove` extends ICON for re-ranking existing concept placements.

## Running the system

There is no build step. The project is used interactively via Jupyter notebooks or by importing directly.

**Demo:** `demo.ipynb` — full walkthrough including model wrapping and a complete ICON run.

**Data wrangling** (run from `experiments/data_wrangling/`, needs `data_config.json` in that directory):
```bash
cd experiments/data_wrangling
python build_emb_data.py   # builds contrastive learning data for emb_model
python build_gen_data.py   # builds seq2seq data for gen_model
python build_sub_data.py   # builds classification data for sub_model
```

**Model training:** Notebooks under `experiments/model_training/` — `train_emb.ipynb`, `train_gen.ipynb`, `train_sub.ipynb`.

**Evaluation:** `experiments/evaluation/knn_eval.ipynb`.

## Code architecture

```
main/
  icon.py          — ICON class: run(), auto(), semiauto(), manual(), outer_loop(), inner_loop(), enhanced_traversal(), insert()
  category_move.py — ICONforCategoryMove(ICON): specialised subclass for concept re-ranking
  config.py        — All config dataclasses (tree_config hierarchy); Update_config() for dynamic config updates

utils/
  taxo_utils.py    — Taxonomy(nx.DiGraph): core graph class with taxonomic operations; TreeTaxonomy subclass; from_json(), from_ontology()
  vector_index.py  — FaissVectorStore: wraps FAISS IVF index for concept embedding search
  tokenset_utils.py — NLP utilities: tokenset(), lemmatization, breadcrumb normalisation, lexical screening helpers
  log_style.py     — ANSI color constants (Fore, Style) for console logging

data/
  raw/             — Source taxonomy files (JSON and OWL)
  emb/, gen/, sub/ — Processed training/eval data per sub-model

experiments/
  data_wrangling/  — Scripts to build training data from raw taxonomy
  model_training/  — Fine-tuning notebooks (BERT for emb/sub, T5 for gen)
  evaluation/      — KNN evaluation notebook
```

## Key design patterns

**Config system:** All ICON parameters are nested frozen dataclasses (`tree_config` hierarchy in `config.py`). `update_config(**kwargs)` resolves the kwarg name via `locate_arg()` and returns a new config via `recursive_replace()`. Config fields are referenced by leaf name, not path — names must be unique across the tree.

**Taxonomy graph:** `Taxonomy` subclasses `nx.DiGraph`. Edges `(u, v)` represent `u subClassOf v`. Node key `0` is always the root. Edge labels (`'original'`, `'auto'`, `'new'`) track provenance. `add_edge` raises `NetworkXError` on cycles. `get_LCA([])` returns bottom nodes; `get_GCD([])` returns top nodes.

**Three operating modes:**
- `auto` — iterates over all bottom-level concepts as seeds; requires all three models
- `semiauto` — user-specified seed list; requires all three models
- `manual` — places user-specified concept labels directly; requires only `sub_model` (optionally `emb_model` for `auto_bases`)

**Search algorithm (`enhanced_traversal`):** Two-stage BFS — top-down for superclasses, bottom-up for subclasses — with `tolerance` controlling how many consecutive misses before pruning. Scores are cached in `_caches.sub_score_cache` keyed by `(sub_label, sup_label)`.

**Vector index:** FAISS IVF index stored per taxonomy identity (`id(taxo)`) in `_caches.vector_store`. Must be rebuilt if the taxonomy object is replaced.

**Lexical check:** Pre-computed hash map `tokenset(label) → node_id` for duplicate detection. Stored per taxonomy identity in `_caches.lexical_cache`.

## Taxonomy JSON format

Input/output format for `from_json()` / `Taxonomy.to_json()`:
```json
{
  "nodes": [{"id": 1, "label": "Electronics"}, ...],
  "edges": [{"src": 42, "tgt": 1, "label": "original"}, ...]
}
```
- `src` is the **child** node ID; `tgt` is the **parent** node ID
- Node `id=0` is reserved for root (auto-created if absent)
- Any extra fields on nodes/edges are preserved as attributes

## Dependencies

Core: `numpy`, `owlready2`, `networkx`, `faiss`, `tqdm`, `nltk`  
Training pipeline: `torch`, `pandas`, `transformers`, `datasets`, `evaluate`, `info-nce-pytorch`  
Python ≥ 3.9 required.
