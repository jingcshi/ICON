import os
import tempfile

import networkx as nx
import pytest

from icon.core.taxonomy import Taxonomy, TreeTaxonomy, from_json


def make_simple_taxo():
    """Root(0) -> Animal(1) -> Mammal(2), Bird(3); Mammal(2) -> Dog(4)"""
    t = Taxonomy()
    for node_id, label in [(0, "Root"), (1, "Animal"), (2, "Mammal"), (3, "Bird"), (4, "Dog")]:
        t.add_node(node_id, label=label)
    for child, parent in [(1, 0), (2, 1), (3, 1), (4, 2)]:
        t.add_edge(child, parent, label="original")
    return t


class TestTaxonomyBasics:

    def test_node_and_edge_count(self):
        t = make_simple_taxo()
        assert t.number_of_nodes() == 5
        assert t.number_of_edges() == 4

    def test_get_label(self):
        t = make_simple_taxo()
        assert t.get_label(0) == "Root"
        assert t.get_label(4) == "Dog"

    def test_get_label_list(self):
        t = make_simple_taxo()
        labels = t.get_label([1, 2])
        assert labels == ["Animal", "Mammal"]

    def test_get_children(self):
        t = make_simple_taxo()
        assert set(t.get_children(1)) == {2, 3}

    def test_get_parents(self):
        t = make_simple_taxo()
        assert t.get_parents(2) == [1]

    def test_get_ancestors(self):
        t = make_simple_taxo()
        assert set(t.get_ancestors(4)) == {2, 1, 0}

    def test_get_descendants(self):
        t = make_simple_taxo()
        assert set(t.get_descendants(1)) == {2, 3, 4}

    def test_get_LCA_leaves(self):
        t = make_simple_taxo()
        leaves = t.get_LCA([])
        assert set(leaves) == {3, 4}

    def test_get_GCD_roots(self):
        t = make_simple_taxo()
        roots = t.get_GCD([])
        assert set(roots) == {0}

    def test_get_LCA_of_subset(self):
        t = make_simple_taxo()
        lca = t.get_LCA([3, 4])
        assert set(lca) == {1}

    def test_subsumes(self):
        t = make_simple_taxo()
        assert t.subsumes(1, 4)    # Animal subsumes Dog
        assert not t.subsumes(3, 4)  # Bird does not subsume Dog

    def test_cycle_prevention(self):
        t = make_simple_taxo()
        with pytest.raises(nx.NetworkXError):
            t.add_edge(0, 4, label="original")  # Root cannot be child of Dog

    def test_add_node_duplicate_key_updates_attr(self):
        t = make_simple_taxo()
        result = t.add_node(1, label="Duplicate")
        assert result == 2  # 2 = node exists, attributes updated
        assert t.get_label(1) == "Duplicate"

    def test_to_and_from_json_roundtrip(self):
        t = make_simple_taxo()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            t.to_json(path)
            t2 = from_json(path)
        finally:
            os.unlink(path)
        assert t2.number_of_nodes() == t.number_of_nodes()
        assert t2.number_of_edges() == t.number_of_edges()
        assert set(t2.nodes) == set(t.nodes)
        for node in t.nodes:
            assert t2.get_label(node) == t.get_label(node)

    def test_reduce_subset(self):
        t = make_simple_taxo()
        # {1, 4}: 1 is ancestor of 4, so reduction keeps 4 (most specific = reverse=True keeps most general)
        reduced = t.reduce_subset([1, 4], reverse=False)
        assert set(reduced) == {4}  # most general: 1 subsumes 4 so 4 is subsumed → keep 1
        # Actually reduce_subset(reverse=False) keeps nodes not subsumed by others → keeps 1 is ambiguous
        # test: reduce_subset(reverse=True) keeps most general
        reduced_r = t.reduce_subset([1, 4], reverse=True)
        assert set(reduced_r) == {1}

    def test_filter_by_level(self):
        t = make_simple_taxo()
        level1 = t.filter_by_level(top_level=0, bottom_level=1)
        assert 0 in level1

    def test_create_insertion_search_space(self):
        t = make_simple_taxo()
        sub = t.create_insertion_search_space([3, 4])
        # subgraph should contain the bases and at least their ancestors
        assert 3 in sub.nodes
        assert 4 in sub.nodes


class TestTreeTaxonomy:

    def test_tree_enforces_single_parent(self):
        t = TreeTaxonomy()
        for node_id, label in [(0, "Root"), (1, "A"), (2, "B"), (3, "C")]:
            t.add_node(node_id, label=label)
        t.add_edge(1, 0, label="original")
        t.add_edge(2, 0, label="original")
        t.add_edge(3, 1, label="original")
        # Second parent should raise NetworkXError
        with pytest.raises(nx.NetworkXError):
            t.add_edge(3, 2, label="original")
