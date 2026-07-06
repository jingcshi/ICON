# Taxonomy File Format

ICON reads and writes taxonomies as JSON with two top-level arrays.

## Schema

```json
{
  "nodes": [
    {"id": 1, "label": "Electronics"},
    {"id": 2, "label": "Smartphones"}
  ],
  "edges": [
    {"src": 2, "tgt": 1, "label": "original"}
  ]
}
```

### Nodes

Each node object must have:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Integer node ID. `0` is reserved for the root and is auto-created if absent — do not use it for user nodes |
| `label` | yes | Surface form / name of the concept |
| *(any other fields)* | no | Stored as node attributes and round-tripped through `Taxonomy` |

### Edges

Each edge object must have:

| Field | Required | Description |
|-------|----------|-------------|
| `src` | yes | Child node ID |
| `tgt` | yes | Parent node ID (`src subClassOf tgt`) |
| `label` | no | Provenance tag. ICON uses `'original'`, `'auto'`, `'new'` |
| *(any other fields)* | no | Stored as edge attributes |

## Loading and saving

```python
from icon import from_json, from_ontology

# Load from JSON file
taxo = from_json("path/to/taxonomy.json")

# Load from OWL ontology (owlready2.Ontology object)
taxo = from_ontology(ontology)

# Save back to JSON
taxo.to_json("path/to/output.json", indent=2)
```

## Notes

- Edges encode `subClassOf`: `src` is the **child**, `tgt` is the **parent**.
- `add_edge` raises `NetworkXError` if the edge would create a cycle.
- Extra node/edge fields survive load → mutate → save round-trips unchanged.
- A full example file is at [`data/raw/google.json`](../data/raw/google.json).
