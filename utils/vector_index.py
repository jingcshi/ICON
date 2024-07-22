import os
import sys
sys.path.append(os.getcwd() + '/..')
from typing import List, Tuple, Union, Hashable, Literal
import numpy as np
import faiss

class FaissVectorStore:
    def __init__(self, 
                vectors: np.ndarray,
                concepts: List[Hashable],
                use_ivf: bool = True,
                nlist: int = None,
                nprobe: int = None,
                metric: Literal[0, 1] = 0, # O for inner product and 1 for L2
                normalize: bool = False) -> None:

        vectors = vectors.astype(np.float32)
        if normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        N = vectors.shape[0]
        D = vectors.shape[1] 
        if use_ivf:
            self.is_ivf = True
            if nlist is None:
                nlist = int(np.sqrt(N))
            if nprobe is None:
                nprobe = int(np.sqrt(nlist))
            index_description = f"IVF{nlist},Flat"
            index = faiss.index_factory(D, index_description, metric)
            index.set_direct_map_type(faiss.DirectMap.Hashtable)
            index.nprobe = nprobe
            index.train(vectors)
        else:
            self.is_ivf = False
            index = faiss.index_factory(D, "IDMap2,Flat", metric)
        index.add_with_ids(vectors, concepts)
        self.index = index
        self.concepts = set(concepts)
        self.ntotal = N
        self.d = D

    def add(self, vectors: np.ndarray, ids: Union[Hashable, List[Hashable]]) -> None:
        if isinstance(ids, Hashable):
            ids = [ids]
            vectors = vectors[np.newaxis]
        self.index.add_with_ids(vectors, ids)
        self.ntotal += len(ids)
        self.concepts = self.concepts.union(ids)
    
    def delete(self, ids: Union[Hashable, List[Hashable]]) -> None:
        if isinstance(ids, Hashable):
            ids = [ids]
        self.ntotal -= self.index.remove_ids(np.array(ids))
        self.concepts = self.concepts.difference(ids)
    
    def search(self, 
               query: np.ndarray,
               k: int = 5,
               subset: List[Hashable] = None,
               nprobe: int = None,
               exhaustive: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        if len(query.shape) == 1:
            single_query = True
            query = query[np.newaxis]
        else:
            single_query = False
        if exhaustive and self.is_ivf:
            nprobe = self.index.nlist
        search_params = {}
        if subset != None:
            search_params['sel'] = faiss.IDSelectorArray(subset)
        if nprobe != None and self.is_ivf:
            search_params['nprobe'] = nprobe
        param_func = faiss.SearchParametersIVF if self.is_ivf else faiss.SearchParameters
        D, I = self.index.search(query, k, params=param_func(**search_params))
        if single_query:
            D = D[0]
            I = I[0]
        return D, I
    
    def reconstruct(self, ids: Union[Hashable, List[Hashable]]) -> np.ndarray:

        if isinstance(ids, Hashable):
            return self.index.reconstruct(ids)
        return self.index.reconstruct_batch(ids)
    
    def retrain(self, vectors: np.ndarray) -> None:
        if self.is_ivf:
            self.index.train(vectors)