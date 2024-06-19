import os
import sys
sys.path.append(os.getcwd() + '/..')
from typing import List, Set, Union, Tuple, Callable, Optional, Hashable, Iterable, Any, Literal
from itertools import combinations
from collections import deque
import re
import torch
import numpy as np
import owlready2 as o2
import networkx as nx
from copy import deepcopy
from tqdm.notebook import tqdm
from utils.log_style import Fore, Style
from utils.taxo_utils import Taxonomy
from utils.tokenset_utils import tokenset
import main.config as _config
import main.icon as _icon

class NullContext:
    '''
    Dummy replacement for a progress bar environment.
    '''
    
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass
    def __bool__(self):
        return False
    def reset(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        pass
    def set_description(self, *args, **kwargs):
        pass

class ICONforCategoryMove(_icon.ICON):
    
    def __init__(self, 
                data: Union[Taxonomy,o2.Ontology]=None,
                ret_model=None,
                gen_model=None,
                sub_model=None,
                mode: Literal['auto', 'manual']='auto',
                method: Literal['search', 'rag']='search',
                rand_seed: Any=114514,
                max_outer_loop: int=None,
                ignore: Union[Iterable[Hashable], re.Pattern]=[],
                input_concepts: Iterable[Hashable]=[],
                retrieve_size: int=10,
                candidate_top_level: int=-1,
                candidate_bottom_level: int=1,
                ret_ignore: Union[Iterable[Hashable], re.Pattern]=[],
                scope_top_level: int=0,
                scope_bottom_level: int=1,
                sub_ignore: Union[Iterable[Hashable], re.Pattern]=[],
                threshold: float=0.5,
                tolerance: int=0,
                force_prune: bool=False,
                always_search_to_bottom : bool=True,
                do_select: bool=True,
                always_include_old: bool = True,
                selection_features: List[Literal['parent', 'siblings']]=['parent', 'siblings'],
                weights: np.ndarray=np.array([1,1], dtype = float),
                do_update: bool=True,
                logging: Union[bool, int, List[str]]=1
                ) -> None:

        if isinstance(data,o2.Ontology):
            data = Taxonomy.from_ontology(data)
        self.data = data
        self.models = _config.icon_models(ret_model,gen_model,sub_model)
        self._caches = _config.icon_caches()
        self._status = _config.icon_status()
        self.config = _config.iconforcategorymove_config(mode=mode,
                                method=method,
                                rand_seed=rand_seed,
                                auto_config=_config.iconforcategorymove_auto_config(max_outer_loop=max_outer_loop, ignore=ignore),
                                manual_config=_config.icon_manual_config(input_concepts),
                                ret_config=_config.iconforcategorymove_ret_config(retrieve_size=retrieve_size, candidate_top_level=candidate_top_level, candidate_bottom_level=candidate_bottom_level, ret_ignore=ret_ignore),
                                sub_config=_config.iconforcategorymove_sub_config(
                                    _config.iconforcategorymove_subgraph_config(scope_top_level=scope_top_level, scope_bottom_level=scope_bottom_level, sub_ignore=sub_ignore),
                                    _config.iconforcategorymove_search_config(threshold=threshold, tolerance=tolerance, force_prune=force_prune, always_search_to_bottom=always_search_to_bottom)),
                                selection_config=_config.iconforcategorymove_selection_config(do_select=do_select, always_include_old=always_include_old, selection_features=selection_features, weights=weights),
                                update_config=_config.icon_update_config(do_update=do_update),
                                logging=logging)
    
    def search(self, taxo: Taxonomy, query: str) -> dict:

        if self.config.sub_config.search.threshold > 1 or self.config.sub_config.search.threshold < 0:
            raise ValueError('Threshold must be in the range [0,1]')
        
        # Stage 1: search for superclasses
        sup = {}
        top = taxo.get_GCD([])
        queue = deque([(n,0) for n in top])
        if top:
            self.update_sub_score_cache([query]*len(top), taxo.get_label(top))
        visited = {}

        while queue:
            node, fails = queue.popleft()
            visited[node] = True
            to_cache = []
            # Key zero is always assumed to be the global root node.
            if node == 0:
                p = 1.0
            else:
                nodelabel = taxo.get_label(node)
                try:
                    p = self._caches.sub_score_cache[(query, nodelabel)]
                except KeyError:
                    p = self.models.sub_model(query, nodelabel).item()
                    self._caches.sub_score_cache[(query, nodelabel)] = p
            
            if p >= self.config.sub_config.search.threshold:
                sup[node] = p
                # Recursively track down the domain to keep searching
                for child in taxo.get_children(node):
                    if child not in visited:
                        queue.append((child,0))
                        to_cache.append(taxo.get_label(child))
                if to_cache:
                    self.update_sub_score_cache([query]*len(to_cache), to_cache)
            elif fails < self.config.sub_config.search.tolerance:
                for child in taxo.get_children(node):
                    if child not in visited:
                        # Keep searching down until success or failures accumulate to tolerance. Used to alleviate misjudgments of the model
                        queue.append((child,fails+1))
                        to_cache.append(taxo.get_label(child))
                if to_cache:
                    self.update_sub_score_cache([query]*len(to_cache), to_cache)
            elif self.config.sub_config.search.force_prune:
                for desc in taxo.get_descendants(node):
                    visited[desc] = True
        
        # At this point, sup consists of all the tested subsumers of query, but there are usually redundancies (non-minimal subsumers). Therefore, we have to remove them
        sup_ancestors = set.union(*[set(taxo.get_ancestors(s)) for s in sup])
        sup = {k:sup[k] for k in set(sup).difference(sup_ancestors)}
        
        # Remove non-leaf categories if specified
        if self.config.sub_config.search.always_search_to_bottom:
            sup = {k:sup[k] for k in set(sup).intersection(taxo.get_LCA([]))}
        
        return sup

    def move(self, target: Hashable, new_parent: Union[List[Hashable], Hashable], old_parent: Optional[Union[List[Hashable], Hashable]] = None) -> None:

        single_new = isinstance(new_parent, Hashable)
        single_old = isinstance(old_parent, Hashable)
        if single_new:
            new_parent = [new_parent]
        if single_old:
            old_parent = [old_parent]

        err = []
        for n in new_parent:
            try:
                self._status.working_taxo.add_edge(target, n, label='moved')
            except nx.NetworkXError:
                err.append(n)
                self.print_log(f'{Fore.RED}{Style.BRIGHT}Warning:{Style.RESET_ALL} Invalid parent {self._status.working_taxo.get_label(n)} ({n}). Skipping.', 3, 'outer_loop_details')
        new_parent = list(set(new_parent).difference(err))
        if old_parent is not None:
            if set(old_parent) == set(new_parent):
                if single_old or len(old_parent) == 1:
                    old_parent = old_parent[0]
                self.print_log(f"{Style.BRIGHT}{Fore.CYAN}{self._status.working_taxo.get_label(target)} {Fore.WHITE}remains{Style.RESET_ALL} under {Style.BRIGHT}{Fore.BLUE}{self._status.working_taxo.get_label(old_parent)}{Style.RESET_ALL} ({old_parent})", 3, "outer_loop_details")
                self._status.progress[0] += 1
                return
        if single_new or len(new_parent) == 1:
            new_parent = new_parent[0]
        self.print_log(f"{Style.BRIGHT}{Fore.CYAN}{self._status.working_taxo.get_label(target)} {Fore.GREEN}moved{Style.RESET_ALL} to under {Style.BRIGHT}{Fore.BLUE}{self._status.working_taxo.get_label(new_parent)}{Style.RESET_ALL} ({new_parent})", 3, "outer_loop_details")
        self._status.progress[1] += 1
        return

    def evaluate_parent(self, query: str, candidate: List[Hashable]) -> np.ndarray:
        '''
        Evaluate the candidate parent by predicting subsumption likelihood between the query and the candidate parent.
        '''
        scores = np.zeros(len(candidate), dtype = float)
        index_to_cache = []
        for i, c in enumerate(candidate):
            if c == 0:
                scores[i] = 1.0
            elif (query, self._status.working_taxo.get_label(c)) in self._caches.sub_score_cache:
                scores[i] = self._caches.sub_score_cache[(query, self._status.working_taxo.get_label(c))]
            else:
                index_to_cache.append(i)
        if index_to_cache:
            self.update_sub_score_cache([query]*len(index_to_cache), self._status.working_taxo.get_label([candidate[i] for i in index_to_cache]))
            for i in index_to_cache:
                scores[i] = self._caches.sub_score_cache[(query, self._status.working_taxo.get_label(candidate[i]))]
        return scores
    
    def evaluate_siblings(self, query: str, candidate: List[Hashable]) -> np.ndarray:
        '''
        Evaluate the candidate parent by measuring the average embedding similarity between the query and its candidate siblings (children of the candidate parent).
        '''
        scores = np.zeros(len(candidate), dtype = float)
        for i, c in enumerate(candidate):
            siblings = self._status.working_taxo.get_children(c)
            if len(siblings) == 0:
                scores[i] = 0.0
            else:
                scores[i] = np.mean(self.models.ret_model.similarity(query, self._status.working_taxo.get_label(siblings)))
        return scores

    def select(self, query: str, candidate: List[Hashable]) -> Hashable:

        if len(self.config.selection_config.selection_features) == 0:
            raise ValueError('No selection feature is specified.')
        if len(self.config.selection_config.selection_features) != self.config.selection_config.weights.shape[0]:
            raise ValueError('The size of weights must match the number of used features.')

        scores = np.zeros((len(candidate), len(self.config.selection_config.selection_features)), dtype = float)
        current_column = 0
        for f in self.config.selection_config.selection_features:
            if f not in ['parent', 'siblings']:
                raise ValueError(f'Invalid selection feature: {f}')
            elif f == 'parent':
                scores[:,current_column] = self.evaluate_parent(query, candidate)
                current_column += 1
            elif f == 'siblings':
                scores[:,current_column] = self.evaluate_siblings(query, candidate)
                current_column += 1
        final_scores = np.dot(scores, self.config.selection_config.weights)
        winner = candidate[np.argmax(final_scores)]
        return [winner]        

    def rag(self, query: str, old_parent: Optional[Union[str, Iterable[str]]]) -> dict:

        
        return {}

    def examine_category(self, target: Hashable) -> None:
            
            if target not in self._status.working_taxo.nodes():
                self.print_log(f'Target {Fore.RED}{Style.BRIGHT}not found{Style.RESET_ALL} in the taxonomy. Skipping.', 2, 'outer_loop')
                return
            if target == 0:
                self.print_log(f'Target is the {Fore.RED}{Style.BRIGHT}root{Style.RESET_ALL} node. Skipping.', 2, 'outer_loop')
                self._status.progress[0] += 1
                return

            # Remove the upwards edges of target from the taxonomy
            old_parent = self._status.working_taxo.get_parents(target)
            for p in old_parent:
                self._status.working_taxo.remove_edge(target, p)
            old_parent_repr = old_parent[0] if len(old_parent) == 1 else old_parent

            # Search for new candidate parents
            if self.config.method == 'search':
                subtaxo = self._status.working_taxo.create_move_search_space(target, self.config.sub_config.subgraph.scope_top_level, self.config.sub_config.subgraph.scope_bottom_level)
                candidates = list(self.search(subtaxo, self._status.working_taxo.get_label(target)))
            elif self.config.method == 'rag':
                candidates = list(self.rag(self._status.working_taxo.get_label(target), self._status.working_taxo.get_label(old_parent)))
            if self.config.selection_config.always_include_old:
                candidates = list(set(candidates).union(set(old_parent)))
            self.print_log(f'Found {Fore.BLACK}{Style.BRIGHT}{len(candidates)}{Style.RESET_ALL} candidate{"" if len(candidates) == 1 else "s"} with {self.config.method}', 3, 'outer_loop_details')
            for c in candidates:
                self.print_log(f'{Fore.BLUE}{Style.BRIGHT}{self._status.working_taxo.get_label(c)}{Style.RESET_ALL} ({c})', 4, 'outer_loop_concept_list')

            # If no candidate is found, keep the target under its original parents
            if not candidates:
                prefix = 'Prediction: ' if not self.config.update_config.do_update else ''
                self.print_log(f'{prefix}{Style.BRIGHT}{Fore.CYAN}{self._status.working_taxo.get_label(target)} {Fore.WHITE}remains{Style.RESET_ALL} under {Style.BRIGHT}{Fore.BLUE}{self._status.working_taxo.get_label(old_parent_repr)}{Style.RESET_ALL} ({old_parent_repr}).', 3, 'outer_loop_details')
                for p in old_parent:
                    self._status.working_taxo.add_edge(target, p, label='moved')
                return
            
            # Select the best candidate if required
            if (self.config.selection_config.do_select or isinstance(self._status.working_taxo, taxo_utils.TreeTaxonomy)) and len(candidates) > 1:
                winner = self.select(self._status.working_taxo.get_label(target), candidates)
            else:
                winner = candidates

            # Update the taxonomy or log the results
            if self.config.update_config.do_update:
                self.move(target, winner, old_parent)
            else:
                if set(winner) == set(old_parent):
                    self.print_log(f"Prediction: {Style.BRIGHT}{Fore.CYAN}{self._status.working_taxo.get_label(target)} {Fore.WHITE}remains{Style.RESET_ALL} under {Style.BRIGHT}{Fore.BLUE}{self._status.working_taxo.get_label(old_parent_repr)}{Style.RESET_ALL} ({old_parent_repr})", 3, "outer_loop_details")
                else:
                    winner_repr = winner[0] if len(winner) == 1 else winner
                    self.print_log(f"Prediction: {Style.BRIGHT}{Fore.CYAN}{self._status.working_taxo.get_label(target)} {Fore.GREEN}moved{Style.RESET_ALL} to under {Style.BRIGHT}{Fore.BLUE}{self._status.working_taxo.get_label(winner_repr)}{Style.RESET_ALL} ({winner_repr})", 3, "outer_loop_details")
            self._status.logs[target] = winner

    def auto(self) -> None:

        leaves = self._status.working_taxo.get_LCA([])
        if self.config.auto_config.ignore:
            if isinstance(self.config.auto_config.ignore, re.Pattern):
                movable = [l for l in leaves if not self.config.auto_config.ignore.match(self._status.working_taxo.get_label(l))]
            else:
                movable = list(set(leaves).difference(self.config.auto_config.ignore))
        total = min(len(movable), self.config.auto_config.max_outer_loop) if self.config.auto_config.max_outer_loop else len(movable)
        self._status.pbar_outer.reset(total = total)
        self._status.pbar_outer.set_description('Auto mode')
        with self._status.pbar_outer:
            for c in movable:
                if self.config.auto_config.max_outer_loop:
                    if self._status.outer_loop_count >= self.config.auto_config.max_outer_loop:
                        self.print_log(f'Iteration limit reached. Exiting.', 1, 'system')
                        break
                self._status.outer_loop_count += 1
                self.print_log(f'Iteration {self._status.outer_loop_count}: Examining category {Fore.CYAN}{Style.BRIGHT}{self._status.working_taxo.get_label(c)}{Style.RESET_ALL} ({c}):', 2, 'outer_loop')
                self.examine_category(c)
                self._status.pbar_outer.update()

    def manual(self) -> None:

        self._status.pbar_outer.reset(total = len(self.config.manual_config.input_concepts))
        self._status.pbar_outer.set_description('Manual mode')
        with self._status.pbar_outer:
            for c in self.config.manual_config.input_concepts:
                self._status.outer_loop_count += 1
                self.print_log(f'Iteration {self._status.outer_loop_count}: Examining category {Fore.CYAN}{Style.BRIGHT}{self._status.working_taxo.get_label(c)}{Style.RESET_ALL} ({c}):', 1, 'outer_loop')
                self.examine_category(c)
                self._status.pbar_outer.update()

    def run(self, **kwargs) -> Union[Taxonomy, dict]:
        
        self._status = _config.icon_status()
        self.update_config(**kwargs)
        if self.config.rand_seed != None:
            np.random.seed(self.config.rand_seed)
            torch.manual_seed(self.config.rand_seed)
        if not self.data:
            raise ValueError('Missing input data')
        self._status.working_taxo = deepcopy(self.data)
        self.print_log(f'Loaded {self.data.__str__()}. Commencing category move', 1, 'system')
        
        self._status.outer_loop_count = 0
        self._status.progress = np.array([0,0], dtype=int)
        progress_bar = (isinstance(self.config.logging,bool) and self.config.logging) or (isinstance(self.config.logging,int) and self.config.logging >= 1) or (isinstance(self.config.logging,list) and 'progress_bar' in self.config.logging)
        if progress_bar:
            self._status.pbar_outer = tqdm(total = 1)
        else:
            self._status.pbar_outer = NullContext()
        
        if self.config.method == 'search':
            if self.models.sub_model is None:
                raise ModuleNotFoundError(f'sub_model is required to run on search method')               
        elif self.config.method == 'rag':
            if self.models.ret_model is None:
                raise ModuleNotFoundError(f'ret_model is required to run on rag method')
            if self.models.sub_model is None:
                raise ModuleNotFoundError(f'sub_model is required to run on rag method')
        else:
            raise ValueError(f'Invalid method: {self.config.method}. Please choose from "search" or "rag"')
        
        if self.config.mode == 'auto':
            self.auto()
        elif self.config.mode == 'manual':
            if self.config.manual_config.input_concepts:
                self.manual()
            else:
                raise ValueError('Manual mode requires input concepts')
        else:
            raise ValueError(f'Invalid mode: {self.config.mode}. Please choose from "auto" or "manual"')

        plural_all = 's' if self._status.progress[0] + self._status.progress[1] > 1 else ''
        plural_keep = 's' if self._status.progress[0] > 1 else ''
        plural_move = 's' if self._status.progress[1] > 1 else ''
        progress_str = f' Examined {Fore.BLACK}{Style.BRIGHT}{self._status.progress[0]+self._status.progress[1]}{Style.RESET_ALL} concept{plural_all}. Moved {Fore.BLACK}{Style.BRIGHT}{self._status.progress[1]}{Style.RESET_ALL} concept{plural_move} and kept {Fore.BLACK}{Style.BRIGHT}{self._status.progress[0]}{Style.RESET_ALL} concept{plural_keep}'
        self.print_log(f'Enrichment complete.{progress_str}.', 1, 'system')
        
        if self.config.update_config.do_update:
            self.print_log(f'Return {self._status.working_taxo.__str__()}', 1, 'system')
            output = self._status.working_taxo
        else:
            self.print_log('Return ICON predictions', 1, 'system')
            output = self._status.logs
        
        return output