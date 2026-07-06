# ICON Configuration Reference

All parameters can be passed as keyword arguments to the `ICON` constructor or updated at any time via `iconobj.update_config(**kwargs)`. Parameter names are unique across the full config tree; use the leaf name directly.

## Global

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | `str` | `'auto'` — enrich entire taxonomy without supervision; `'semiauto'` — enrich using user-specified seeds; `'manual'` — place user-specified concept labels directly (no `gen_model` required) |
| `logging` | `int` / `bool` / `list` | `0`/`False` suppresses all output; `1` shows progress bar and brief updates; `True`/`5` shows everything; integers 2–5 for intermediate verbosity; a list of message type strings for selective logging |
| `rand_seed` | `int` or `None` | Passed to NumPy and PyTorch for reproducibility |
| `transitive_reduction` | `bool` | Whether to perform transitive reduction on the output taxonomy, removing redundant edges |

## Auto mode

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_outer_loop` | `int` | Maximum number of outer loops allowed |

## Semiauto mode

| Parameter | Type | Description |
|-----------|------|-------------|
| `semiauto_seeds` | `Iterable` | Concept identifiers to use as seeds, one per outer loop |

## Manual mode

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_concepts` | `Iterable[str]` | New concept labels to place in the taxonomy |
| `manual_concept_bases` | `Iterable` or `None` | If provided, each entry becomes the search bases for the corresponding input concept |
| `auto_bases` | `bool` | If `True`, ICON builds search bases automatically using `emb_model`. Disabling allows `manual` mode without `emb_model` |

## Retrieval

| Parameter | Type | Description |
|-----------|------|-------------|
| `retrieve_size` | `int` | Number of concepts to retrieve per query |
| `restrict_combinations` | `bool` | Restrict candidate subsets to those containing the seed concept |

## Generation

| Parameter | Type | Description |
|-----------|------|-------------|
| `ignore_label` | `set[str]` | Labels that indicate `gen_model` rejection; generated concepts with these labels are discarded |
| `filter_subsets` | `bool` | Skip subsets whose LCA is trivial (i.e., the LCA is within the subset itself) |

## Concept placement — search domain

| Parameter | Type | Description |
|-----------|------|-------------|
| `subgraph_crop` | `bool` | Limit search domain to descendants of the LCAs of the search bases |
| `subgraph_force` | `list[list[str]]` or `None` | Force the search domain to always include LCAs of the search bases with respect to sub-taxonomies defined by edge label sets. No effect if `subgraph_crop=False` |
| `subgraph_strict` | `bool` | Further restrict search domain to subsumers of at least one base concept |

## Concept placement — search

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | `float` | Minimum `sub_model` probability to accept a subsumption |
| `tolerance` | `int` | Maximum depth to continue searching a rejected branch before pruning |
| `force_known_subsumptions` | `bool` | Force placement to be at least as general as the LCA of search bases and at least as specific as their union; also stops search at the search bases |
| `force_prune_branches` | `bool` | Force rejection of all subclasses of a tested non-superclass (and vice versa). Slows search on tree-like taxonomies |

## Taxonomy update

| Parameter | Type | Description |
|-----------|------|-------------|
| `do_update` | `bool` | If `True`, return the enriched taxonomy; if `False`, return a predictions dictionary |
| `eqv_score_func` | `callable` | Function `(p_sub, p_sup) -> float` estimating equivalence probability from two subsumption scores. Default: multiplication |
| `do_lexical_check` | `bool` | Pre-compute a lexical hash map at init and use it to screen new concepts for duplicates |

## Example

```python
from icon import ICON

icon = ICON(
    data=your_taxonomy,
    emb_model=your_emb_model,
    gen_model=your_gen_model,
    sub_model=your_sub_model,
    mode='auto',
    threshold=0.8,
    tolerance=2,
    logging=1,
)
icon.run()

# Update config at any time
icon.update_config(threshold=0.9, ignore_label=icon.config.gen_config.ignore_label | {'Owl:Thing'})
icon.run()
```
