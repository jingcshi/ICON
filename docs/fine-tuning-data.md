# Fine-Tuning Data Reference

The `/experiments/data_wrangling/data_config.json` file controls all data generation scripts. Run scripts from that directory:

```bash
cd experiments/data_wrangling
python build_emb_data.py   # contrastive learning data for emb_model
python build_gen_data.py   # seq2seq data for gen_model
python build_sub_data.py   # binary classification data for sub_model
```

## Universal parameters

Applied across all three scripts.

| Parameter | Description |
|-----------|-------------|
| `random_seed` | NumPy seed for reproducibility (omit for non-deterministic) |
| `data_path` | Path to the raw taxonomy file |
| `eval_split_rate` | Fraction of data reserved for evaluation; range `[0, 1)` |

## EMB model parameters

Produces `(query, positive, negative_1, …, negative_k)` contrastive tuples. Positives are siblings in the taxonomy; negatives are dissimilar concepts.

| Parameter | Description |
|-----------|-------------|
| `concept_appearance_per_file` | How many times each concept appears across the generated data |
| `negative_per_minibatch` | Number of negatives per tuple (`k` above) |

Sample output: [`data/emb/google-eval.csv`](../data/emb/google-eval.csv)

## GEN model parameters

Produces `([PREFIX]C1;…;Cn, LCA)` pairs. The model learns to predict a concept's least common ancestor given a list of related concepts. Some rows are optionally corrupted so the LCA degenerates to the root.

| Parameter | Description |
|-----------|-------------|
| `max_chunk_size` | Maximum list length (≥ 2); data includes all lengths from 1 to this value |
| `corrupt_ratio` | Fraction of rows to corrupt; range `[0, 1]` |
| `corrupt_patterns` | List of `(p, n)` integer pairs: `p` uncorrupted concepts + `n` random concepts. Requires `p + n ≤ max_chunk_size` and `p ≠ 1` |
| `pattern_weight` | Relative frequency for each corrupt pattern (same length as `corrupt_patterns`; need not sum to 1) |
| `prompt_prefix` | Task prefix prepended to all concept lists |

Sample output: [`data/gen/google-eval.csv`](../data/gen/google-eval.csv)

## SUB model parameters

Produces `(sub, sup, ref)` triples where `ref=1` iff `sub` is a sub-concept of `sup`. Positives are child-parent and grandchild-grandparent pairs. Negatives are generated in two modes.

| Parameter | Description |
|-----------|-------------|
| `easy_negative_sample_rate` | Number of easy negatives relative to positives. Obtained by replacing `sup` with a random concept |
| `hard_negative_sample_rate` | Number of hard negatives relative to positives. Obtained by replacing `sup` with a concept reachable via random walk from the original `sup` |

Sample output: [`data/sub/google-eval.csv`](../data/sub/google-eval.csv)
