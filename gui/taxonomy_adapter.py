"""Thin wrapper around Taxonomy that tracks dirty state and an undo stack."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from collections import deque
from dataclasses import dataclass
from typing import Optional
from icon.core.taxonomy import Taxonomy, from_json, from_owl


@dataclass
class _Snapshot:
    taxo: Taxonomy
    description: str


class TaxonomyAdapter:
    MAX_UNDO = 20

    def __init__(self):
        self._taxo: Optional[Taxonomy] = None
        self._path: Optional[str] = None
        self._dirty: bool = False
        self._undo_stack: deque[_Snapshot] = deque(maxlen=self.MAX_UNDO)

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.json':
            taxo = from_json(path)
        elif ext in ('.owl', '.rdf', '.ttl', '.n3'):
            taxo = from_owl(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        self._taxo = taxo
        self._path = path
        self._dirty = False
        self._undo_stack.clear()

    def save(self, path: Optional[str] = None) -> None:
        dest = path or self._path
        if dest is None:
            raise ValueError("No path specified")
        self._taxo.to_json(dest)
        self._path = dest
        self._dirty = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def taxo(self) -> Optional[Taxonomy]:
        return self._taxo

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def loaded(self) -> bool:
        return self._taxo is not None

    # ── Edit helpers ───────────────────────────────────────────────────────────

    def _checkpoint(self, description: str) -> None:
        import copy
        self._undo_stack.append(_Snapshot(copy.deepcopy(self._taxo), description))
        self._dirty = True

    def undo(self) -> Optional[str]:
        """Restore previous state; returns description or None if nothing to undo."""
        if not self._undo_stack:
            return None
        snap = self._undo_stack.pop()
        self._taxo = snap.taxo
        self._dirty = True
        return snap.description

    def rename_node(self, node_id, label: str) -> None:
        self._checkpoint(f"rename {node_id}")
        self._taxo.set_label(node_id, label)

    def add_node(self, node_id, label: str, parent_id) -> None:
        self._checkpoint(f"add node {node_id}")
        self._taxo.add_node(node_id, label=label)
        self._taxo.add_edge(node_id, parent_id, label='original')

    def delete_node(self, node_id) -> None:
        self._checkpoint(f"delete {node_id}")
        self._taxo.remove_node(node_id)

    def add_edge(self, child_id, parent_id) -> None:
        self._checkpoint(f"add edge {child_id}->{parent_id}")
        self._taxo.add_edge(child_id, parent_id, label='original')

    def remove_edge(self, child_id, parent_id) -> None:
        self._checkpoint(f"remove edge {child_id}->{parent_id}")
        self._taxo.remove_edge(child_id, parent_id)

    # ── Subgraph for canvas ────────────────────────────────────────────────────

    def neighborhood(self, node_id, radius: int = 2):
        """Return a subgraph centered on node_id up to `radius` hops (ancestor + descendant)."""
        taxo = self._taxo
        visible = {node_id}
        # ancestors up to radius
        frontier = {node_id}
        for _ in range(radius):
            next_f = set()
            for n in frontier:
                next_f.update(taxo.get_parents(n))
            visible.update(next_f)
            frontier = next_f
        # descendants up to radius
        frontier = {node_id}
        for _ in range(radius):
            next_f = set()
            for n in frontier:
                next_f.update(taxo.get_children(n))
            visible.update(next_f)
            frontier = next_f
        return taxo.subgraph(visible)
