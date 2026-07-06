import numpy as np

from icon.utils.vector_index import FaissVectorStore


def make_store(n=100, dim=64):
    """Use n=100 to avoid FAISS IVF 'too few training points' warning."""
    np.random.seed(42)
    vecs = np.random.rand(n, dim).astype(np.float32)
    concepts = list(range(n))
    return FaissVectorStore(vecs, concepts), vecs, concepts


class TestFaissVectorStore:

    def test_build_and_size(self):
        store, _, _ = make_store(n=20)
        assert len(store.concepts) == 20

    def test_search_returns_k_results(self):
        store, vecs, _ = make_store(n=20)
        scores, indices = store.search(vecs[0:1], k=5, exhaustive=True)
        assert indices.shape == (1, 5)

    def test_search_top1_is_self(self):
        store, vecs, concepts = make_store(n=20)
        _, indices = store.search(vecs[0:1], k=1, exhaustive=True)
        assert indices[0, 0] == concepts[0]

    def test_reconstruct(self):
        store, vecs, _ = make_store(n=20)
        rec = store.reconstruct(0)
        assert rec.shape == (1, 64)
        np.testing.assert_allclose(rec[0], vecs[0], atol=1e-5)

    def test_add_and_search(self):
        store, vecs, _ = make_store(n=20)
        new_vec = np.ones((1, 64), dtype=np.float32)
        store.add(new_vec, 999)
        assert 999 in store.concepts
        assert len(store.concepts) == 21

    def test_delete(self):
        store, _, _ = make_store(n=20)
        store.delete(0)
        assert 0 not in store.concepts
        assert len(store.concepts) == 19

    def test_search_with_subset(self):
        store, vecs, _ = make_store(n=20)
        subset = [5, 6, 7, 8, 9]
        scores, indices = store.search(vecs[5:6], k=3, subset=subset, exhaustive=True)
        for idx in indices.flatten():
            assert idx in subset
