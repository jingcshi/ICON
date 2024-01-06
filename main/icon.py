import os
import sys
sys.path.append(os.getcwd() + '/..')
from typing import List, Set, Union, Tuple, Callable, Optional, Hashable, Iterable, Any, Literal
from itertools import combinations
from collections import deque
import torch
import numpy as np
import owlready2 as o2
import networkx as nx
from tqdm.notebook import tqdm
from colorama import Fore, Style
from utils.taxo_utils import Taxonomy
from utils.breadcrumb import tokenset
import utils.config as _Config

class NullContext:
    
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass
    def __bool__(self):
        return False

    '''
    Self-supervised taxonomy enrichment system.
    This system uses a seed class to retrieve a cluster of closely related classes, in order to zoom in on a small facet of the taxonomy. It enumerates subsets of the cluster and uses a text generative model to create a virtual concept that is expected to represent the common parent for each subset. The generated concept will go through a series of valiadations and its placement in the taxonomy will be decided by a search based on pairwise subsumption prediction. The outcome for each validated concept will be either a new class inserted to the taxonomy, or a merger with existing class(es). The taxonomy is being updated dynamically each step. The cycle can be applied iteratively with different seeds to perform more comprehensive enrichment.

    The algorithm depends on three (and one optional) plug-in models:
    KNN_model(taxonomy, Class) -> List[Class]: Retrieve the classes most closely related with the input class in the input taxonomy
    GEN_model(List[str]) -> str: Generate the common parent label for an arbitrary set of class labels
    SUB_model(List[Class],List[Class]) -> float: Predict whether one class subsumes another, and should support batching

    Args:
        Required:
            data: The taxonomy to enrich. If an ontology is given, the taxonomy will be created from it.
            knn_model, gen_model, sub_model: Plugin models as described above
        Options:
            Overall control
                mode: Select one of the following
                    'auto': The system will automatically enrich the entire taxonomy without supervision
                    'semiauto': The system will enrich the taxonomy with the seeds specified by user input
                    'manual': The system will try to place the new concepts specified by user input directly into the taxonomy
                max_outer_loop: Maximal number of cycles allowed. Used for auto mode.
                semiauto_seeds: An iterable of classes in the taxonomy from which to kick off enrichment. Used for semiauto mode.
                manual_inputs: An iterable of new concept labels to be placed in the taxonomy. Used for manual mode.
                rand_seed: Leave untouched or set a number to ensure reproducibility. Set to None to disable.
            Class retrieval
                retrieve_size: Number of relevant classes to retrieve from the KNN model
                restrict_combinations: Whether to limit the class combinations to only those including the seed class
            Label generation
                ignore_label: The set of outputs which indicate the GEN model's rejection to generate a common parent label
            Search domain
                subgraph_crop: Whether to limit the search domain to the descendants of the LCA of the classes which are used to generate the new concept (henceforth referred to as the base classes)
                subgraph_force: If provided (type: list of list of labels), the search domain will always include the LCA of base classes w.r.t. the sub-taxonomy defined by the edges whose labels are in each list of the input. Will not take effect if subgraph_crop = False
                subgraph_strict: Whether to further limit the search domain to the subsumers of base classes
                subs_threshold: The model's minimal predicted probability for accepting subsumption
            Search
                search_tolerance: Maximal depth to continue searching a branch that has been rejected by subsumption test before pruning branch
                force_known_subsumptions: Whether to force the search to place the new class at least as general as the LCA of the seed classes used by GEN model, and at least as specific as the union of the seed classes 
                    Enabling this will also force the search to stop at the seed classes
                force_prune_branches: Whether to force the search to reject all subclasses of a tested non-superclass in superclass search, and to reject all superclasses of a tested non-subclass in subclass search
                    Enabling this will slow down the search if the taxonomy is roughly tree-like
                eqv_score_func: If the model predicts that query both subsumes and is subsumed by another class, this function will be used to calculate the overall likelihood of equivalence given the two likelihoods for each subsumption. Default is the product of two likelihoods.
            Post-processing
                transitive_reduction: Whether to perform transitive reduction on the enriched taxonomy (all the taxonomy's original data will be preserved)
            Logging
                log: Setting this on will allow the algorithm to print its procedure at the level specified. Useful for visualisation and debugging. Logging >= 1 will enable progress bar display. More details on logging available at the docstring of print_log
    '''
    
class ICON:
    
    def __init__(self,
                data: Union[Taxonomy,o2.Ontology]=None,
                ret_model=None,
                gen_model=None,
                sub_model=None,
                lexical_cache: dict={},
                sub_score_cache: dict={},
                mode: Literal['auto', 'semiauto', 'manual']='auto',
                max_outer_loop: int=None,
                semiauto_seeds: List[Hashable]=[],
                input_concepts: List[str]=[],
                input_concept_bases: List[List[Hashable]]=None,
                rand_seed: Any=114514,
                retrieve_size: int=10,
                restrict_combinations: bool=True,
                ignore_label: List[str]=['','All categories','Root Concept','Thing','Allcats','Everything','root'],
                filter_subset: bool=True,
                subgraph_crop: bool=True,
                subgraph_force: List[List[str]]=[['original']],
                subgraph_strict: bool=True,
                threshold: float=0.5,
                tolerance: int=0,
                force_base_subsumptions: bool=False,
                force_prune: bool=False,
                do_update: bool=True,
                eqv_score_func: Callable[[Tuple[float, float]], float]=lambda x: x[0]*x[1],
                do_lexical_check: bool=True,
                transitive_reduction: bool=True,
                log: Union[bool, int, List[str]]=False,
                ) -> None:
        
        if isinstance(data,o2.Ontology):
            data = Taxonomy.from_ontology(data)
        self.data = data
        self.models = _Config.icon_models(ret_model,gen_model,sub_model)
        self.caches = _Config.icon_caches(lexical_cache,sub_score_cache)
        self.status = _Config.icon_status()
        self.config = _Config.icon_config(mode,
                                rand_seed,
                                _Config.icon_auto_config(max_outer_loop),
                                _Config.icon_semiauto_config(semiauto_seeds),
                                _Config.icon_manual_config(input_concepts,input_concept_bases),
                                _Config.icon_ret_config(retrieve_size,restrict_combinations),
                                _Config.icon_gen_config(ignore_label,filter_subset),
                                _Config.icon_sub_config(
                                    _Config.icon_subgraph_config(subgraph_crop,subgraph_force,subgraph_strict),
                                    _Config.icon_search_config(threshold,tolerance,force_base_subsumptions,force_prune)),
                                _Config.icon_update_config(do_update,eqv_score_func,do_lexical_check),
                                transitive_reduction,
                                log)
        
        if data != None and do_lexical_check:
            self.load_lexical_cache(data)
        
    def load_lexical_cache(self, data: Taxonomy) -> None:
        
        with tqdm(total = data.number_of_nodes(), leave=False, desc='Loading lexical cache') as pbar:
            for n,l in data.nodes(data='label'):
                # The dictionary used for NIL entity check. Keys are hash values of tokenised and lemmatised class labels
                self.caches.lexical_cache[hash(tuple(tokenset(l)))] = n
                pbar.update()
    
    def update_lexical_cache(self, node: Hashable, label: str) -> None:
        
        self.caches.lexical_cache[hash(tuple(tokenset(label)))] = node
    
    def clear_lexical_cache(self) -> None:
        
        self.caches.lexical_cache = {}
    
    def lexical_check(self, label: str) -> Optional[Hashable]:
        
        try:
            return self.caches.lexical_cache[hash(tuple(tokenset(label)))]
        except KeyError:
            return None
    
    def update_sub_score_cache(self, sub: List[str], sup: List[str]) -> None:
        
        # The model output is expected to be a numpy.array of shape [len(keypair)]
        outputs = self.models.sub_model(sub, sup)
        for i,s in enumerate(outputs):
            self.caches.sub_score_cache[(sub[i], sup[i])] = s.item()
    
    def clear_sub_score_cache(self) -> None:
        
        self.caches.sub_score_cache = {}
    
    def print_log(self, msg: str, level: int, msgtype: str) -> Literal[0,1]:
        
        setting = self.config.log
        if (isinstance(setting,bool) and setting == True) or (isinstance(setting,int) and setting >= level) or (isinstance(setting,list) and msgtype in setting):
            indent = max(level-1, 0)
            print('\t' * indent + msg)
            return 1
        return 0

    def update_config(self, **kwargs) -> None:
        
        for arg, value in kwargs.items():
            self.config = _Config.Update_config(self.config, arg, value)

    def run(self, **kwargs) -> Taxonomy:
        
        self.update_config(**kwargs)
        if self.config.rand_seed != None:
            np.random.seed(self.config.rand_seed)
            torch.manual_seed(self.config.rand_seed)
        if not self.data:
            raise ValueError('Missing input data')
        self.status.working_taxo = self.data.copy()    
        self.print_log(f'Loaded {self.data.__str__()}. Commencing enrichment', 1, 'system')
        
        self.status.outer_loop_count = 0
        self.status.progress = np.array([0,0], dtype=int)
        self.status.nextkey = max(hash(n) for n in self.status.working_taxo.nodes)+1 # Track the next ID in case of new class insertion
        progress_bar = (isinstance(self.config.log,bool) and self.config.log) or (isinstance(self.config.log,int) and self.config.log >= 1) or (isinstance(self.config.log,list) and 'progress_bar' in self.config.log)
        if progress_bar:
            self.status.pbar_outer = tqdm(total = 1, position = 0)
            self.status.pbar_inner = tqdm(total = 1, position = 1, leave=False) if self.config.mode in ['auto', 'semiauto'] else NullContext()
        else:
            self.status.pbar_outer = NullContext()
            self.status.pbar_inner = NullContext()
        
        # Sample a random untouched bottom class as seed each cycle, or use the given seeds if provided
        if self.config.mode == 'auto':
            for model in self.models.leaf_fields():
                if getattr(self.models, model) is None:
                    raise ModuleNotFoundError(f'{model} is required to run auto mode')
            self.auto()
                            
        elif self.config.mode == 'semiauto':
            for model in self.models.leaf_fields():
                if getattr(self.models, model) is None:
                    raise ModuleNotFoundError(f'{model} is required to run semiauto mode')
            self.semiauto()
                            
        else:
            if self.models.sub_model is None:
                raise ModuleNotFoundError(f'sub_model is required to run manual mode')
            self.manual()
        
        suffix = ' with transitive reduction' if self.config.transitive_reduction else ''
        progress_str = f' Added {Fore.BLACK}{Style.BRIGHT}{self.status.progress[0]}{Style.RESET_ALL} new classes and {Fore.BLACK}{Style.BRIGHT}{self.status.progress[1]}{Style.RESET_ALL} new direct subsumptions.' if self.config.update_config.do_update else ''
        self.print_log(f'Enrichment complete.{progress_str} Begin post-processing{suffix}', 1, 'system')
        if self.config.transitive_reduction:
            tr = nx.transitive_reduction(self.status.working_taxo)
            tr.add_nodes_from(self.status.working_taxo.nodes(data=True))
            tr.add_edges_from((u, v, self.status.working_taxo.edges[u, v]) for u, v in tr.edges)
            self.status.working_taxo = Taxonomy(tr)
        self.status.working_taxo.add_edges_from((u, v, self.data.edges[u, v]) for u, v in self.data.edges)
        
        self.print_log(f'Return {self.status.working_taxo.__str__()}', 1, 'system')
        return self.status.working_taxo
    
    def auto(self, **kwargs) -> None:
        
        self.update_config(**kwargs)
        seedpool = self.status.working_taxo.get_LCA([],return_type=set)
        poolsize = len(seedpool)
        if not self.config.auto_config.max_outer_loop:
            max_outer_loop = poolsize
        else:
            max_outer_loop = self.config.auto_config.max_outer_loop
        if self.status.pbar_outer:
            self.status.pbar_outer.reset(total=poolsize)
            self.status.pbar_outer.set_description('Auto mode')
            
        with self.status.pbar_outer:
            with self.status.pbar_inner:
                while self.status.outer_loop_count < max_outer_loop and seedpool:
                    candidates = list(seedpool)
                    poolsize = len(candidates)
                    seed = candidates[np.random.choice(poolsize,1).item()]
                    self.print_log(f'Outer loop {Fore.BLACK}{Style.BRIGHT}{self.status.outer_loop_count}{Style.RESET_ALL}: Seed {seed} ({Fore.BLUE}{Style.BRIGHT}{self.status.working_taxo.get_label(seed)}{Style.RESET_ALL}) selected from {Fore.BLACK}{Style.BRIGHT}{poolsize}{Style.RESET_ALL} possible candidates', 2, 'cycle')
                    self.status.outer_loop_count += 1
                    outer_loop_progress, processed = self.outer_loop(seed)
                    self.status.progress += outer_loop_progress
                    seedpool = seedpool.difference(processed)
                    if self.status.pbar_outer:
                        new_poolsize = len(seedpool)
                        diff = poolsize - new_poolsize
                        poolsize = new_poolsize
                        self.status.pbar_outer.update(diff)
                        
    def semiauto(self, **kwargs) -> None:
        
        self.update_config(**kwargs)
        if not self.config.semiauto_config.semiauto_seeds:
                raise ValueError('Please provide a list of seeds in semiauto mode')
        if self.status.pbar_outer:
            self.status.pbar_outer.reset(total=len(self.config.semiauto_config.semiauto_seeds))
            self.status.pbar_outer.set_description('Semiauto mode')
            
        with self.status.pbar_outer:
            with self.status.pbar_inner:
                for seed in self.config.semiauto_config.semiauto_seeds:
                    self.print_log(f'Ouer loop {Fore.BLACK}{Style.BRIGHT}{self.status.outer_loop_count}{Style.RESET_ALL}: Seed {seed} ({Fore.BLUE}{Style.BRIGHT}{self.status.working_taxo.get_label(seed)}{Style.RESET_ALL})', 2, 'cycle')
                    self.status.outer_loop_count += 1
                    outer_loop_progress, _ = self.outer_loop(seed)
                    self.status.progress += outer_loop_progress
                    if self.status.pbar_outer:
                        self.status.pbar_outer.update()

    def manual(self, **kwargs) -> None:
        
        self.update_config(**kwargs)
        if not self.config.manual_config.input_concepts:
            raise ValueError('Please provide a list of manual inputs in manual mode')
        if not self.config.manual_config.input_concept_bases:
            inputs_bases = [[]] * len(self.config.manual_config.input_concepts)
        elif len(self.config.manual_config.input_concepts) != len(self.config.manual_config.input_concept_bases):
            raise ValueError('Lengths of input_concepts and input_concept_bases must match')
        else:
            inputs_bases = self.config.manual_config.input_concept_bases
        if self.status.pbar_outer:
            self.status.pbar_outer.reset(total = len(self.config.manual_config.input_concepts))
            self.status.pbar_outer.set_description('Manual mode')
            
        with self.status.pbar_outer:
            for i, newlabel in enumerate(self.config.manual_config.input_concepts):
                self.print_log(f'Input: {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL}', 3, 'iter')
                self.status.outer_loop_count += 1
                inner_loop_progress = self.inner_loop(newlabel, inputs_bases[i])
                self.status.progress += inner_loop_progress
                if self.status.pbar_outer:
                    self.status.pbar_outer.update()
    
    def generate(self, base: Iterable[Hashable]) -> Optional[str]:
    
        # Skip if the LCA is already in base
        if self.status.working_taxo.get_LCA(base,return_type=set).issubset(set(base)):
            self.print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Skipped{Style.RESET_ALL} because a trivial LCA exists', 3, 'iter')
            return None

        # Generate new CP label
        newlabel = self.models.gen_model([self.status.working_taxo.get_label(b) for b in base])

        # Reject if the generated label is considered root
        if newlabel in self.config.gen_config.ignore_label:
            self.print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Rejected{Style.RESET_ALL} by label generator', 3, 'iter')
            return None

        self.print_log(f'Generated common parent label: {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL}', 3, 'iter')
        return newlabel
    
    def enhanced_traversal(self, taxo: Taxonomy, newlabel: str, base: Iterable[Hashable]) -> Tuple[dict, dict, dict]:
        '''
        Search for the optimal placement (equivalences, superclasses and subclasses) of a new concept (represented by text) in a given taxonomy
        The basic algorithm is a two-stage BFS, one top-down in search for superclasses and one bottom-up in search for subclasses

        Args:
            taxo: The search domain. Does not have to be the full taxonomy
            newlabel: Concept to insert
            model: Subsumption prediction model. Should accept input params in the shape of (subclass:str,superclass:str)
            base: The seed classes (represented by keys) used to generate the newlabel. Has effect only if force_known=True
            threshold: The model's minimal predicted probability for accepting subsumption, should be in range [0,1]
            tolerance: Maximal depth to continue searching a branch that has been rejected by subsumption test before pruning branch
            force_known: Whether to force the new class to be no more general than the LCA of base, and no more specific than the union of base
            force_prune: Whether to force the search to reject all subclasses of a tested non-superclass in superclass search, and to reject all superclasses of a tested non-subclass in subclass search
                Enabling this will slow down the search if the taxonomy is roughly tree-like

        Output:
            sup, sub, eqv: The model's expected optimal superclasses, subclasses and equivalent classes of query
            All returns are dictionaries where the keys are class keys and values are relation likelihoods
                If sup=None, the query has been rejected by search
                Else if eqv!=None, the query has been mapped to the eqv classes by search
                Else, the query has been accepted as a new class by search
        '''
        if self.config.sub_config.search.threshold > 1 or self.config.sub_config.search.threshold < 0:
            raise ValueError('Threshold must be in the range [0,1]')
        
        force_known = self.config.sub_config.search.force_base_subsumptions and base
        # Stage 1: search for superclasses
        sup = {}
        # If force_known=True, the starting point of this search becomes the LCA of the base w.r.t. the original taxonomy, plus any other LCA. Otherwise, the starting point is get_GCD(empty set) which returns the top nodes in the domain
        if force_known:
            top = taxo.get_LCA(base, return_type=set).union(taxo.get_LCA(base, labels='original',return_type=set))
            top = taxo.reduce_subset(top, reverse=True)
        else:
            top = taxo.get_GCD([])
        queue = deque([(n,0) for n in top])
        if top:
            self.update_sub_score_cache([newlabel]*len(top), taxo.get_label(top))
        visited = {}

        while queue:
            node, fails = queue.popleft()
            visited[node] = True
            to_cache = []
            # Key zero is always assumed to be the global root node. If force_known we also force the search to respect known subsumptions.
            if node == 0 or (force_known and all([taxo.subsumes(node, b) for b in base])):
                p = 1.0
            else:
                nodelabel = taxo.get_label(node)
                try:
                    p = self.caches.sub_score_cache[(newlabel, nodelabel)]
                except KeyError:
                    p = self.models.sub_model(newlabel, nodelabel).item()
                    self.caches.sub_score_cache[(newlabel, nodelabel)] = p
            
            if p >= self.config.sub_config.search.threshold:
                sup[node] = p
                if force_known and (node in base):
                    # In the superclass search stage, the base classes mark the end of search on the current branch (the query should never be more specific than the base). It is possible, however, that some of the base classes subsume the query.
                    if self.config.sub_config.search.force_prune:
                        # Mark all descendants as non-subsumers. Same below
                        for desc in taxo.get_descendants(node):
                            visited[desc] = True
                    continue
                # Recursively track down the domain to keep searching
                for child in taxo.get_subclasses(node):
                    if child not in visited:
                        queue.append((child,0))
                        to_cache.append(taxo.get_label(child))
                if to_cache:
                    self.update_sub_score_cache([newlabel]*len(to_cache), to_cache)
            elif fails < self.config.sub_config.search.tolerance:
                for child in taxo.get_subclasses(node):
                    if child not in visited:
                        # Keep searching down until success or failures accumulate to tolerance. Used to alleviate misjudgments of the model
                        queue.append((child,fails+1))
                        to_cache.append(taxo.get_label(child))
                if to_cache:
                    self.update_sub_score_cache([newlabel]*len(to_cache), to_cache)
            elif self.config.sub_config.search.force_prune:
                for desc in taxo.get_descendants(node):
                    visited[desc] = True
        
        # Reject the query
        if not sup:
            return {}, {}, {}
        
        # At this point, sup consists of all the tested subsumers of query, but there are usually redundancies (non-minimal subsumers). Therefore, we have to remove them
        sup_ancestors = set.union(*[set(taxo.get_ancestors(s)) for s in sup])
        sup = {k:sup[k] for k in set(sup).difference(sup_ancestors)}
        
        # Stage 2: search for subclasses
        sub = {}
        eqv = {}
        # get_LCA(empty set) returns the bottom nodes in the domain, which we use as starting point of this search
        bottom = taxo.get_LCA([])
        queue = deque([(n,0) for n in bottom])
        if bottom:
            self.update_sub_score_cache(taxo.get_label(bottom), [newlabel]*len(bottom))
        # The redundant superclasses are also logically certain to be non-subclasses, so we do not search on them
        visited = {k:True for k in sup_ancestors}
        
        while queue:
            node, fails = queue.popleft()
            visited[node] = True
            to_cache = []
            # Force the search to respect known subsumptions
            if force_known and node in base:
                p = 1.0
            else:
                nodelabel = taxo.get_label(node)
                try:
                    p = self.caches.sub_score_cache[(nodelabel, newlabel)]
                except KeyError:
                    p = self.models.sub_model(nodelabel, newlabel).item()
                    self.caches.sub_score_cache[(nodelabel, newlabel)] = p
                
            if p >= self.config.sub_config.search.threshold:
                if node in sup:
                    # A class is both superclass and subclass, this is when we add an equivalence instead
                    # The equivalence likelihood is stored as tuple (sup_likelihood, sub_likelihood)
                    eqv[node] = (sup.pop(node),p)
                    continue
                else:
                    sub[node] = p
                # Recursively track up the domain to keep searching
                for parent in taxo.get_superclasses(node):
                    if parent not in visited:
                        queue.append((parent,0))
                        to_cache.append(taxo.get_label(parent))
                if to_cache:
                    self.update_sub_score_cache(to_cache, [newlabel]*len(to_cache))
            elif fails < self.config.sub_config.search.tolerance:
                # Keep searching up until success or failures accumulate to tolerance
                for parent in taxo.get_superclasses(node):
                    if parent not in visited:
                        queue.append((parent,fails+1))
                        to_cache.append(taxo.get_label(parent))
                if to_cache:
                    self.update_sub_score_cache(to_cache, [newlabel]*len(to_cache))
            elif self.config.sub_config.search.force_prune:
                for ance in taxo.get_ancestors(node):
                    visited[ance] = True
        
        # Same story as in stage 1, we only keep the maximal subclasses
        if sub:
            sub = {k:sub[k] for k in taxo.reduce_subset(list(sub.keys()),reverse=True)}
        
        return sup, sub, eqv
    
    def update_taxonomy(self, taxo:Taxonomy, new:str, eqv: Optional[Hashable], sup: Optional[list], sub: Optional[list]) -> np.ndarray:
        '''
        Insert or merge a *new* concept (represented by its label) into taxonomies, and add corresponding subsumptions. Newly inserted edges will be labelled 'added' in the companion taxonomies
        All input classes should be represented by their keys

        Args:
            taxo: The taxonomy to update
            new: The new concept to be dealt with
            newkey: The key to be used to identify the newly inserted class. If not provided then default to merging with an existing class (which requires eqv)
            eqv: An existing class which the new concept should be equivalent with. If not provided then default to inserting new class (which requires newkey)
            sup, sub: Lists which the new class is supposed to be subsumed by / subsume

        Return the count of new class / direct subsumptions in a np.array([class_count,subsumption_count])
        '''
        additions = np.array([0,0], dtype=int)
        sup_set = set()
        sub_set = set()
        superr_set = set()
        suberr_set = set()

        # Clean up the superclass / subclass sets. We only want to add the most specific superclasses and most general subclasses
        sup = taxo.reduce_subset(sup)
        sub = taxo.reduce_subset(sub, reverse=True)
            
        # Case 1: Merge with a known equivalent class    
        if eqv:
            if eqv in taxo.nodes:
                self.print_log(f'Declared {Fore.MAGENTA}{Style.BRIGHT}equivalence{Style.RESET_ALL} between {Fore.YELLOW}{Style.BRIGHT}{taxo.get_label(eqv)}{Style.RESET_ALL} ({eqv}) and {Fore.CYAN}{Style.BRIGHT}{new}{Style.RESET_ALL}', 4, 'iter_details')
                selfclass = eqv
                self_color = Fore.YELLOW
                self_label = taxo.get_label(eqv)
            else:
                raise KeyError(f'Equivalent class {eqv} not found')
            
        # Case 2: Add a new class
        else:
            if taxo.add_node(self.status.nextkey,label=new) == 0:
                self.print_log(f'{Fore.GREEN}{Style.BRIGHT}Created{Style.RESET_ALL} new class {Fore.CYAN}{Style.BRIGHT}{new}{Style.RESET_ALL} with key {Fore.BLACK}{Style.BRIGHT}{self.status.nextkey}{Style.RESET_ALL}', 4, 'iter_details')
                self.update_lexical_cache(self.status.nextkey, new)
                selfclass = self.status.nextkey
                self_color = Fore.CYAN
                self_label = new
                self.status.nextkey += 1
                additions[0] = 1
            else:
                raise KeyError(f'Key conflict: {self.status.nextkey}')
        
        # Update taxonomies
        for superclass in sup:
            try:
                if taxo.add_edge(selfclass,superclass,label='new') == 0:
                    sup_set.add(superclass)
            except TypeError:
                superr_set.add(superclass)
        for subclass in sub:
            try:
                if taxo.add_edge(subclass,selfclass,label='new') == 0:
                    sub_set.add(subclass)
            except TypeError:
                suberr_set.add(subclass)
        additions[1] = len(sup_set) + len(sub_set)
        
        # Log actions
        if sup_set:
            verb = 'are' if len(sup_set) > 1 else 'is'
            suffix = 'es' if len(sup_set) > 1 else ''
            self.print_log(f'The following class{suffix} {verb} declared as {Fore.MAGENTA}{Style.BRIGHT}superclass{suffix}{Style.RESET_ALL} of {self_color}{Style.BRIGHT}{self_label}{Style.RESET_ALL}:', 5, 'iter_classlist')
            for supc in sup_set:
                self.print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(supc)} {Style.RESET_ALL}({supc})', 5, 'iter_classlist')
        if sub_set:
            verb = 'are' if len(sub_set) > 1 else 'is'
            suffix = 'es' if len(sub_set) > 1 else ''
            self.print_log(f'The following class{suffix} {verb} declared as {Fore.MAGENTA}{Style.BRIGHT}subclass{suffix}{Style.RESET_ALL} of {self_color}{Style.BRIGHT}{self_label}{Style.RESET_ALL}:', 5, 'iter_classlist')
            for subc in sub_set:
                self.print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(subc)} {Style.RESET_ALL}({subc})', 5, 'iter_classlist')

        err_set = superr_set.union(suberr_set)
        if err_set:
            verb = 'are' if len(err_set) > 1 else 'is'
            suffix = 's' if len(err_set) > 1 else ''
            self.print_log(f'{Fore.RED}{Style.BRIGHT}Warning{Style.RESET_ALL}: The following subClassOf relation{suffix} {verb} {Fore.BLACK}{Style.BRIGHT}discarded{Style.RESET_ALL} because of cyclic inheritance:', 5, 'iter_classlist')
            for supc in superr_set:
                self.print_log(f'\t{self_color}{Style.BRIGHT}{self_label} {Fore.BLACK}--> {Fore.BLUE}{taxo.get_label(supc)} {Style.RESET_ALL}({supc})', 5, 'iter_classlist')
            for subc in suberr_set:
                    self.print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(subc)} {Style.RESET_ALL}({subc}) {Fore.BLACK}{Style.BRIGHT}--> {self_color}{self_label}{Style.RESET_ALL}', 5, 'iter_classlist')
        
        return additions
    
    def outer_loop(self, seed: Hashable) -> Tuple[np.ndarray, Set[Hashable]]:
        
        outer_loop_progress = np.array([0,0],dtype=int)
    
        # Retrieve a set of relevant classes from the KNN model, henceforce referred to as base_classes
        base_classes = self.models.ret_model(self.status.working_taxo,seed,k=self.config.ret_config.retrieve_size)
        self.print_log(f'Retrieved {Fore.BLACK}{Style.BRIGHT}{len(base_classes)}{Style.RESET_ALL} classes', 3, 'cycle_details')
        for c in base_classes:
            self.print_log(f'{Fore.BLUE}{Style.BRIGHT}{self.status.working_taxo.get_label(c)}{Style.RESET_ALL}', 4, 'cycle_classlist')
        
        # Use base class pairs as prompt for the GEN model to generate new class names
        if self.config.ret_config.restrict_combinations:
            non_seed = base_classes.copy()
            non_seed.remove(seed)
            intermediate_concepts = [(seed, b) for b in non_seed]
        else:
            intermediate_concepts = list(combinations(base_classes, 2))
        
        if self.status.pbar_inner:
            self.status.pbar_inner.reset(total = len(intermediate_concepts))
            self.status.pbar_inner.set_description(f'Outer loop {self.status.outer_loop_count}')
        for i,subset in enumerate(intermediate_concepts):
            msg = f'Inner loop {Fore.BLACK}{Style.BRIGHT}{self.status.outer_loop_count}.{i+1}{Style.RESET_ALL}: Combination ('
            for b in subset:
                msg += f'{Fore.BLUE}{Style.BRIGHT}{self.status.working_taxo.get_label(b)}{Style.RESET_ALL}, '
            self.print_log(msg[:-2] + ')', 3, 'iter')
            
            newlabel = self.generate(subset)
            if newlabel:
                inner_loop_progress = self.inner_loop(newlabel, subset)
                outer_loop_progress += inner_loop_progress
            if self.status.pbar_inner:
                self.status.pbar_inner.update()
        
        progress_str = f' Added {Fore.BLACK}{Style.BRIGHT}{outer_loop_progress[0]}{Style.RESET_ALL} new classes and {Fore.BLACK}{Style.BRIGHT}{outer_loop_progress[1]}{Style.RESET_ALL} new direct subsumptions.' if self.config.update_config.do_update else ''
        self.print_log(f'Outer loop complete.{progress_str}', 2, 'cycle')
        return outer_loop_progress, set(base_classes)
    
    def inner_loop(self, newlabel: str, base: Iterable[Hashable]) -> np.ndarray:
    
        # Create search domain.
        subtaxo = self.status.working_taxo.create_subgraph(base, crop_top=self.config.sub_config.subgraph.subgraph_crop, force_labels=self.config.sub_config.subgraph.subgraph_force, strict=self.config.sub_config.subgraph.subgraph_strict)
        subgraph_size = len(subtaxo.nodes)
        self.print_log(f'Searching on a domain of {subgraph_size} classes', 4, 'iter_details')

        # Search for optimal placement using the SUB model to predict subsumption.
        sup, sub, eqv = self.enhanced_traversal(subtaxo,newlabel,base)

        resolution = self.lexical_check(newlabel) if self.config.update_config.do_lexical_check else None
        if resolution:
            # We give 100% confidence to the linkage of NIL model, thus making it supercede any other possible equivalences provided by search
            eqv[resolution] = (1.0,1.0)
            self.print_log(f'\tSearch complete. {Fore.YELLOW}{Style.BRIGHT}Mapped{Style.RESET_ALL} to a known class by lexical check', 3, 'iter')
        else:
            self.print_log(f'\tSearch complete. {Fore.GREEN}{Style.BRIGHT}Validated{Style.RESET_ALL} by lexical check', 3, 'iter')

        # Reject the new class because there is no good placement
        if not sup and not eqv:
            self.print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Rejected{Style.RESET_ALL} by search because no good placement can be found', 3, 'iter')
            return np.array([0,0], dtype=int)
        # When there are more than one equivalent classes, keep only the most confident equivalent class
        # Demote the other equivalent classes to either superclasses or subclasses, whichever got the higher likelihood
        if len(eqv) > 1:
            ranked_eqvclasses = [k for k,_ in sorted(eqv.items(), key=lambda x: self.config.update_config.eqv_score_func(x[1]), reverse=True)]
            do_nil_string = ' / lexical check' if resolution else ''
            self.print_log(f'{Fore.RED}{Style.BRIGHT}Warning{Style.RESET_ALL}: Search{do_nil_string} suggests that {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL} is {Fore.MAGENTA}{Style.BRIGHT}equivalent{Style.RESET_ALL} to multiple known classes', 4, 'iter_details')
            for eqvclass in ranked_eqvclasses:
                prob = self.config.update_config.eqv_score_func(eqv[eqvclass])
                self.print_log(f'{Fore.BLUE}{Style.BRIGHT}{self.status.working_taxo.get_label(eqvclass)}{Style.RESET_ALL} with score {prob:.4f}', 5, 'iter_classlist')
            for eqvclass in ranked_eqvclasses[1:]:
                if eqv[eqvclass][0] >= eqv[eqvclass][1]:
                    sup[eqvclass] = eqv.pop(eqvclass)[0]
                else:
                    sub[eqvclass] = eqv.pop(eqvclass)[1]
            self.print_log(f'For safety, only the highest ranked equivalence is preserved', 4, 'iter_details')

        if eqv:
            eqvc = list(eqv)[0]
            sup.pop(eqvc, None)
            sub.pop(eqvc, None)
            eqv = eqvc
            if not resolution:
                self.print_log(f'\t{Fore.YELLOW}{Style.BRIGHT}Mapped{Style.RESET_ALL} to a known class by search', 3, 'iter')
        else:
            eqv = None
            self.print_log(f'\t{Fore.GREEN}{Style.BRIGHT}Accepted{Style.RESET_ALL} as a new class by search', 3, 'iter')
        
        return self.update_taxonomy(self.status.working_taxo,newlabel,eqv=eqv,sup=list(sup),sub=list(sub)) if self.config.update_config.do_update else np.array([0,0], dtype=int)
