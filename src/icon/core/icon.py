from collections import deque
from copy import deepcopy
from itertools import combinations
from typing import Any, Callable, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import owlready2 as o2
import torch
from tqdm.auto import tqdm

import icon.config.config as _config
from icon.core.taxonomy import Taxonomy
from icon.utils.log_style import Fore, Style
from icon.utils.tokenset_utils import tokenset
from icon.utils.vector_index import FaissVectorStore


class NullContext:
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

class ICON:
    '''
    A self-supervised taxonomy enrichment system designed for implicit taxonomy completion.
    '''

    def __init__(self,
                data: Union[Taxonomy,o2.Ontology]=None,
                emb_model=None,
                gen_model=None,
                sub_model=None,
                mode: Literal['auto', 'semiauto', 'manual']='auto',
                max_outer_loop: int=None,
                semiauto_seeds: List[Hashable]=[],
                input_concepts: List[str]=[],
                manual_concept_bases: List[List[Hashable]]=None,
                auto_bases: bool=False,
                rand_seed: Any=114514,
                retrieve_size: int=10,
                restrict_combinations: bool=True,
                ignore_label: List[str]=['','All categories','Root Concept','Thing','Allcats','Everything','root'],
                filter_subset: bool=True,
                subgraph_crop: bool=True,
                subgraph_force: List[List[str]]=[['auto', 'original']],
                subgraph_strict: bool=True,
                threshold: float=0.5,
                tolerance: int=0,
                force_base_subsumptions: bool=False,
                force_prune: bool=False,
                do_update: bool=True,
                eqv_score_func: Callable[[Tuple[float, float]], float]=lambda x: x[0]*x[1],
                do_lexical_check: bool=True,
                transitive_reduction: bool=True,
                logging: Union[bool, int, List[str]]=1
                ) -> None:

        if isinstance(data,o2.Ontology):
            data = Taxonomy.from_ontology(data)
        self.data = data
        self.models = _config.icon_models(emb_model,gen_model,sub_model)
        self._caches = _config.icon_caches()
        self._status = _config.icon_status()
        self.config = _config.icon_config(mode,
                                rand_seed,
                                _config.icon_auto_config(max_outer_loop),
                                _config.icon_semiauto_config(semiauto_seeds),
                                _config.icon_manual_config(input_concepts,manual_concept_bases,auto_bases),
                                _config.icon_ret_config(retrieve_size,restrict_combinations),
                                _config.icon_gen_config(ignore_label,filter_subset),
                                _config.icon_sub_config(
                                    _config.icon_subgraph_config(subgraph_crop,subgraph_force,subgraph_strict),
                                    _config.icon_search_config(threshold,tolerance,force_base_subsumptions,force_prune)),
                                _config.icon_update_config(do_update,eqv_score_func,do_lexical_check),
                                transitive_reduction,
                                logging)

        if self.data is not None and do_lexical_check:
            self.load_lexical_cache(self.data)

    def load_lexical_cache(self, taxo: Taxonomy, data: dict=None) -> None:

        if data is None:
            self._caches.lexical_cache[id(taxo)] = {}
            with tqdm(total = taxo.number_of_nodes(), leave=False, desc='Loading lexical cache') as pbar:
                for n, label in taxo.nodes(data='label'):
                    self._caches.lexical_cache[id(taxo)][hash(tuple(tokenset(label)))] = n
                    pbar.update()
        else:
            self._caches.lexical_cache[id(taxo)] = deepcopy(data)

    def update_lexical_cache(self, taxo: Taxonomy, node: Hashable, label: str) -> None:

        self._caches.lexical_cache[id(taxo)][hash(tuple(tokenset(label)))] = node

    def clear_lexical_cache(self, taxo: Taxonomy=None) -> None:

        if taxo is None:
            self._caches.lexical_cache = {}
        else:
            self._caches.lexical_cache.pop(id(taxo), None)

    def lexical_check(self, taxo: Taxonomy, label: str) -> Optional[Hashable]:

        try:
            return self._caches.lexical_cache[id(taxo)][hash(tuple(tokenset(label)))]
        except KeyError:
            return None

    def check_vector_index_available(self, taxo: Taxonomy, replace_if_found: bool=True) -> bool:

        for vs in self._caches.vector_store:
            if self._caches.vector_store[vs].concepts == set(taxo.nodes):
                if replace_if_found:
                    self._caches.vector_store[id(taxo)] = self._caches.vector_store.pop(vs)
                return True
        return False

    def build_vector_index(self, taxo: Taxonomy) -> None:

        concepts = list(taxo.nodes())
        sentences = self.models.emb_model(taxo.get_label(concepts))
        self._caches.vector_store[id(taxo)] = FaissVectorStore(sentences, concepts)

    def add_concepts_to_vector_index(self, taxo: Taxonomy, c: Union[Hashable, List[Hashable]]) -> None:

        sentence = self.models.emb_model(taxo.get_label(c))
        self._caches.vector_store[id(taxo)].add(sentence, c)

    def remove_concepts_from_vector_index(self, taxo: Taxonomy, c: Union[Hashable, List[Hashable]]) -> None:

        self._caches.vector_store[id(taxo)].delete(c)

    def clear_vector_index(self, taxo: Taxonomy=None) -> None:

        if taxo is None:
            self._caches.vector_store = {}
        else:
            self._caches.vector_store.pop(id(taxo), None)

    def update_sub_score_cache(self, sub: List[str], sup: List[str]) -> None:

        outputs = self.models.sub_model(sub, sup)
        for i,s in enumerate(outputs):
            self._caches.sub_score_cache[(sub[i], sup[i])] = s.item()

    def clear_sub_score_cache(self) -> None:

        self._caches.sub_score_cache = {}

    def print_log(self, msg: str, level: int, msgtype: str, newline = True) -> Literal[0,1]:

        setting = self.config.logging
        if (isinstance(setting,bool) and setting) or (isinstance(setting,int) and setting >= level) or (isinstance(setting,list) and msgtype in setting):
            indent = max(level-1, 0)
            print('\t' * indent + msg, end = '\n' if newline else '')

    def update_config(self, **kwargs) -> None:

        for arg, value in kwargs.items():
            self.config = _config.Update_config(self.config, arg, value)

    def generate(self, base: Iterable[Hashable]) -> Optional[str]:

        if self.config.gen_config.filter_subset:
            if self._status.working_taxo.get_LCA(base,return_type=set).issubset(set(base)):
                self.print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Skipped{Style.RESET_ALL} because a trivial LCA exists', 3, 'inner_loop')
                return None

        newlabel = self.models.gen_model([self._status.working_taxo.get_label(b) for b in base])

        if newlabel in self.config.gen_config.ignore_label:
            self.print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Rejected{Style.RESET_ALL} by label generator', 3, 'inner_loop')
            return None

        self.print_log(f'Generated semantic union label: {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL}', 3, 'inner_loop')
        return newlabel

    def enhanced_traversal(self, taxo: Taxonomy, newlabel: str, base: Iterable[Hashable]) -> Tuple[dict, dict, dict]:

        if self.config.sub_config.search.threshold > 1 or self.config.sub_config.search.threshold < 0:
            raise ValueError('Threshold must be in the range [0,1]')

        force_known = self.config.sub_config.search.force_base_subsumptions and base
        sup = {}
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
            if node == 0 or (force_known and all([taxo.subsumes(node, b) for b in base])):
                p = 1.0
            else:
                nodelabel = taxo.get_label(node)
                try:
                    p = self._caches.sub_score_cache[(newlabel, nodelabel)]
                except KeyError:
                    p = self.models.sub_model(newlabel, nodelabel).item()
                    self._caches.sub_score_cache[(newlabel, nodelabel)] = p

            if p >= self.config.sub_config.search.threshold:
                sup[node] = p
                if force_known and (node in base):
                    if self.config.sub_config.search.force_prune:
                        for desc in taxo.get_descendants(node):
                            visited[desc] = True
                    continue
                for child in taxo.get_children(node):
                    if child not in visited:
                        queue.append((child,0))
                        to_cache.append(taxo.get_label(child))
                if to_cache:
                    self.update_sub_score_cache([newlabel]*len(to_cache), to_cache)
            elif fails < self.config.sub_config.search.tolerance:
                for child in taxo.get_children(node):
                    if child not in visited:
                        queue.append((child,fails+1))
                        to_cache.append(taxo.get_label(child))
                if to_cache:
                    self.update_sub_score_cache([newlabel]*len(to_cache), to_cache)
            elif self.config.sub_config.search.force_prune:
                for desc in taxo.get_descendants(node):
                    visited[desc] = True

        if not sup:
            return {}, {}, {}

        sup_ancestors = set.union(*[set(taxo.get_ancestors(s)) for s in sup])
        sup = {k:sup[k] for k in set(sup).difference(sup_ancestors)}

        sub = {}
        eqv = {}
        bottom = taxo.get_LCA([])
        queue = deque([(n,0) for n in bottom])
        if bottom:
            self.update_sub_score_cache(taxo.get_label(bottom), [newlabel]*len(bottom))
        visited = {k:True for k in sup_ancestors}

        while queue:
            node, fails = queue.popleft()
            visited[node] = True
            to_cache = []
            if force_known and node in base:
                p = 1.0
            else:
                nodelabel = taxo.get_label(node)
                try:
                    p = self._caches.sub_score_cache[(nodelabel, newlabel)]
                except KeyError:
                    p = self.models.sub_model(nodelabel, newlabel).item()
                    self._caches.sub_score_cache[(nodelabel, newlabel)] = p

            if p >= self.config.sub_config.search.threshold:
                if node in sup:
                    eqv[node] = (sup.pop(node),p)
                    continue
                else:
                    sub[node] = p
                for parent in taxo.get_parents(node):
                    if parent not in visited:
                        queue.append((parent,0))
                        to_cache.append(taxo.get_label(parent))
                if to_cache:
                    self.update_sub_score_cache(to_cache, [newlabel]*len(to_cache))
            elif fails < self.config.sub_config.search.tolerance:
                for parent in taxo.get_parents(node):
                    if parent not in visited:
                        queue.append((parent,fails+1))
                        to_cache.append(taxo.get_label(parent))
                if to_cache:
                    self.update_sub_score_cache(to_cache, [newlabel]*len(to_cache))
            elif self.config.sub_config.search.force_prune:
                for ance in taxo.get_ancestors(node):
                    visited[ance] = True

        if sub:
            sub = {k:sub[k] for k in taxo.reduce_subset(list(sub.keys()),reverse=True)}

        return sup, sub, eqv

    def insert(self, taxo:Taxonomy, new:str, eqv: Optional[Hashable], sup: Optional[list], sub: Optional[list]) -> np.ndarray:

        additions = np.array([0,0], dtype=int)
        sup_set = set()
        sub_set = set()
        superr_set = set()
        suberr_set = set()

        sup = taxo.reduce_subset(sup)
        sub = taxo.reduce_subset(sub, reverse=True)

        if eqv:
            if eqv in taxo.nodes:
                self.print_log(f'Declared {Fore.MAGENTA}{Style.BRIGHT}equivalence{Style.RESET_ALL} between {Fore.YELLOW}{Style.BRIGHT}{taxo.get_label(eqv)}{Style.RESET_ALL} ({eqv}) and {Fore.CYAN}{Style.BRIGHT}{new}{Style.RESET_ALL}', 4, 'inner_loop_details')
                selfclass = eqv
                self_color = Fore.YELLOW
                self_label = taxo.get_label(eqv)
            else:
                raise KeyError(f'Equivalent class {eqv} not found')
        else:
            if taxo.add_node(self._status.nextkey,label=new) == 0:
                self.print_log(f'{Fore.GREEN}{Style.BRIGHT}Created{Style.RESET_ALL} new class {Fore.CYAN}{Style.BRIGHT}{new}{Style.RESET_ALL} with key {Fore.BLACK}{Style.BRIGHT}{self._status.nextkey}{Style.RESET_ALL}', 4, 'inner_loop_details')
                self.update_lexical_cache(self._status.working_taxo, self._status.nextkey, new)
                selfclass = self._status.nextkey
                self_color = Fore.CYAN
                self_label = new
                self._status.nextkey += 1
                additions[0] = 1
            else:
                raise KeyError(f'Key conflict: {self._status.nextkey}')

        for superclass in sup:
            try:
                if taxo.add_edge(selfclass,superclass,label='new') == 0:
                    sup_set.add(superclass)
            except nx.NetworkXError:
                superr_set.add(superclass)
        for subclass in sub:
            try:
                if taxo.add_edge(subclass,selfclass,label='new') == 0:
                    sub_set.add(subclass)
            except nx.NetworkXError:
                suberr_set.add(subclass)
        additions[1] = len(sup_set) + len(sub_set)

        if sup_set:
            verb = 'are' if len(sup_set) > 1 else 'is'
            suffix = 'es' if len(sup_set) > 1 else ''
            self.print_log(f'The following class{suffix} {verb} declared as {Fore.MAGENTA}{Style.BRIGHT}superclass{suffix}{Style.RESET_ALL} of {self_color}{Style.BRIGHT}{self_label}{Style.RESET_ALL}:', 5, 'inner_loop_concept_list')
            for supc in sup_set:
                self.print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(supc)} {Style.RESET_ALL}({supc})', 5, 'inner_loop_concept_list')
        if sub_set:
            verb = 'are' if len(sub_set) > 1 else 'is'
            suffix = 'es' if len(sub_set) > 1 else ''
            self.print_log(f'The following class{suffix} {verb} declared as {Fore.MAGENTA}{Style.BRIGHT}subclass{suffix}{Style.RESET_ALL} of {self_color}{Style.BRIGHT}{self_label}{Style.RESET_ALL}:', 5, 'inner_loop_concept_list')
            for subc in sub_set:
                self.print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(subc)} {Style.RESET_ALL}({subc})', 5, 'inner_loop_concept_list')

        err_set = superr_set.union(suberr_set)
        if err_set:
            verb = 'are' if len(err_set) > 1 else 'is'
            suffix = 's' if len(err_set) > 1 else ''
            self.print_log(f'{Fore.RED}{Style.BRIGHT}Warning{Style.RESET_ALL}: The following subClassOf relation{suffix} {verb} {Fore.BLACK}{Style.BRIGHT}discarded{Style.RESET_ALL} because of cyclic inheritance:', 5, 'inner_loop_concept_list')
            for supc in superr_set:
                self.print_log(f'\t{self_color}{Style.BRIGHT}{self_label} {Fore.BLACK}--> {Fore.BLUE}{taxo.get_label(supc)} {Style.RESET_ALL}({supc})', 5, 'inner_loop_concept_list')
            for subc in suberr_set:
                    self.print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(subc)} {Style.RESET_ALL}({subc}) {Fore.BLACK}{Style.BRIGHT}--> {self_color}{self_label}{Style.RESET_ALL}', 5, 'inner_loop_concept_list')

        return additions

    def inner_loop(self, newlabel: str, base: Iterable[Hashable]) -> np.ndarray:

        subtaxo = self._status.working_taxo.create_insertion_search_space(base, crop_top=self.config.sub_config.subgraph.subgraph_crop, force_labels=self.config.sub_config.subgraph.subgraph_force, strict=self.config.sub_config.subgraph.subgraph_strict)
        subgraph_size = len(subtaxo.nodes)
        self.print_log(f'Searching on a domain of {subgraph_size} classes', 4, 'inner_loop_details')

        sup, sub, eqv = self.enhanced_traversal(subtaxo,newlabel,base)

        resolution = self.lexical_check(self._status.working_taxo, newlabel) if self.config.update_config.do_lexical_check else None
        if resolution:
            eqv[resolution] = (1.0,1.0)
            self.print_log(f'\tSearch complete. {Fore.YELLOW}{Style.BRIGHT}Mapped{Style.RESET_ALL} to a known class by lexical check', 3, 'inner_loop')
        else:
            self.print_log(f'\tSearch complete. {Fore.GREEN}{Style.BRIGHT}Validated{Style.RESET_ALL} by lexical check', 3, 'inner_loop')

        if not sup and not eqv:
            self.print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Rejected{Style.RESET_ALL} by search because no good placement can be found', 3, 'inner_loop')
            return np.array([0,0], dtype=int)

        if len(eqv) > 1:
            ranked_eqvclasses = [k for k,_ in sorted(eqv.items(), key=lambda x: self.config.update_config.eqv_score_func(x[1]), reverse=True)]
            do_nil_string = ' or lexical check' if resolution else ''
            self.print_log(f'{Fore.RED}{Style.BRIGHT}Warning{Style.RESET_ALL}: Search{do_nil_string} suggests that {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL} is {Fore.MAGENTA}{Style.BRIGHT}equivalent{Style.RESET_ALL} to multiple known classes', 4, 'inner_loop_details')
            for eqvclass in ranked_eqvclasses:
                prob = self.config.update_config.eqv_score_func(eqv[eqvclass])
                self.print_log(f'{Fore.BLUE}{Style.BRIGHT}{self._status.working_taxo.get_label(eqvclass)}{Style.RESET_ALL} with score {prob:.4f}', 5, 'inner_loop_concept_list')
            for eqvclass in ranked_eqvclasses[1:]:
                if eqv[eqvclass][0] >= eqv[eqvclass][1]:
                    sup[eqvclass] = eqv.pop(eqvclass)[0]
                else:
                    sub[eqvclass] = eqv.pop(eqvclass)[1]
            self.print_log('For safety, only the highest ranked equivalence is preserved', 4, 'inner_loop_details')

        if eqv:
            eqvc = list(eqv)[0]
            eqv[eqvc] = self.config.update_config.eqv_score_func(eqv[eqvc])
            sup.pop(eqvc, None)
            sub.pop(eqvc, None)
            if not resolution:
                self.print_log(f'\t{Fore.YELLOW}{Style.BRIGHT}Mapped{Style.RESET_ALL} to a known class by search', 3, 'inner_loop')
        else:
            eqvc = None
            self.print_log(f'\t{Fore.GREEN}{Style.BRIGHT}Accepted{Style.RESET_ALL} as a new class by search', 3, 'inner_loop')

        self._status.logs[newlabel] = {'equivalent': eqv, 'superclass': sup, 'subclass': sub}
        return self.insert(self._status.working_taxo,newlabel,eqv=eqvc,sup=list(sup),sub=list(sub)) if self.config.update_config.do_update else np.array([0,0], dtype=int)

    def outer_loop(self, seed: Hashable) -> Tuple[np.ndarray, Set[Hashable]]:

        outer_loop_progress = np.array([0,0],dtype=int)

        vstore = self._caches.vector_store[id(self._status.working_taxo)]
        _, base_classes = vstore.search(vstore.reconstruct(seed), k=self.config.ret_config.retrieve_size, exhaustive=True)
        base_classes = base_classes.tolist()
        self.print_log(f'Retrieved {Fore.BLACK}{Style.BRIGHT}{len(base_classes)}{Style.RESET_ALL} classes', 3, 'outer_loop_details')
        for c in base_classes:
            self.print_log(f'{Fore.BLUE}{Style.BRIGHT}{self._status.working_taxo.get_label(c)}{Style.RESET_ALL}', 4, 'outer_loop_concept_list')

        if self.config.ret_config.restrict_combinations:
            non_seed = base_classes.copy()
            non_seed.remove(seed)
            intermediate_concepts = [(seed, b) for b in non_seed]
        else:
            intermediate_concepts = list(combinations(base_classes, 2))

        if self._status.pbar_inner and intermediate_concepts:
            self._status.pbar_inner.reset(total = len(intermediate_concepts))
            self._status.pbar_inner.set_description(f'Outer loop {self._status.outer_loop_count}')
        for i,subset in enumerate(intermediate_concepts):
            msg = f'Inner loop {Fore.BLACK}{Style.BRIGHT}{self._status.outer_loop_count}.{i+1}{Style.RESET_ALL}: Combination ('
            for b in subset:
                msg += f'{Fore.BLUE}{Style.BRIGHT}{self._status.working_taxo.get_label(b)}{Style.RESET_ALL}, '
            self.print_log(msg[:-2] + ')', 3, 'inner_loop')

            newlabel = self.generate(subset)
            if newlabel:
                inner_loop_progress = self.inner_loop(newlabel, subset)
                outer_loop_progress += inner_loop_progress
            if self._status.pbar_inner:
                self._status.pbar_inner.update()

        if self.config.update_config.do_update:
            self.print_log(f'Outer loop complete. Added {Fore.BLACK}{Style.BRIGHT}{outer_loop_progress[0]}{Style.RESET_ALL} new classes and {Fore.BLACK}{Style.BRIGHT}{outer_loop_progress[1]}{Style.RESET_ALL} new direct subsumptions.', 2, 'outer_loop')
        return outer_loop_progress, set(base_classes)

    def auto(self, **kwargs) -> None:

        self.update_config(**kwargs)
        if self.check_vector_index_available(self._status.working_taxo):
            self.print_log('Found pre-computed vector index', 1, 'system')
        else:
            self.print_log('Building vector index...', 1, 'system', newline=False)
            self.build_vector_index(self._status.working_taxo)
            self.print_log('Complete', 1, 'system')
        seedpool = self._status.working_taxo.get_LCA([],return_type=set)
        poolsize = len(seedpool)
        if not self.config.auto_config.max_outer_loop:
            max_outer_loop = poolsize
        else:
            max_outer_loop = self.config.auto_config.max_outer_loop
        if self._status.pbar_outer:
            self._status.pbar_outer.reset(total=poolsize)
            self._status.pbar_outer.set_description('Auto mode')

        with self._status.pbar_outer:
            with self._status.pbar_inner:
                while self._status.outer_loop_count < max_outer_loop and seedpool:
                    candidates = list(seedpool)
                    poolsize = len(candidates)
                    seed = candidates[np.random.choice(poolsize,1).item()]
                    self._status.outer_loop_count += 1
                    plural_str = 's' if poolsize > 1 else ''
                    self.print_log(f'Outer loop {Fore.BLACK}{Style.BRIGHT}{self._status.outer_loop_count}{Style.RESET_ALL}: Seed {seed} ({Fore.BLUE}{Style.BRIGHT}{self._status.working_taxo.get_label(seed)}{Style.RESET_ALL}) selected from {poolsize} possible candidate{plural_str}', 2, 'outer_loop')
                    outer_loop_progress, processed = self.outer_loop(seed)
                    self._status.progress += outer_loop_progress
                    seedpool = seedpool.difference(processed)
                    if self._status.pbar_outer:
                        new_poolsize = len(seedpool)
                        diff = poolsize - new_poolsize
                        poolsize = new_poolsize
                        self._status.pbar_outer.update(diff)

    def semiauto(self, **kwargs) -> None:

        self.update_config(**kwargs)
        if not self.config.semiauto_config.semiauto_seeds:
                raise ValueError('Please provide a list of seeds in semiauto mode')
        if self._status.pbar_outer:
            self._status.pbar_outer.reset(total=len(self.config.semiauto_config.semiauto_seeds))
            self._status.pbar_outer.set_description('Semiauto mode')

        if self.check_vector_index_available(self._status.working_taxo):
            self.print_log('Found pre-computed vector index', 1, 'system')
        else:
            self.print_log('Building vector index...', 1, 'system', newline=False)
            self.build_vector_index(self._status.working_taxo)
            self.print_log('Complete', 1, 'system')
        with self._status.pbar_outer:
            with self._status.pbar_inner:
                for seed in self.config.semiauto_config.semiauto_seeds:
                    self._status.outer_loop_count += 1
                    self.print_log(f'Outer loop {Fore.BLACK}{Style.BRIGHT}{self._status.outer_loop_count}{Style.RESET_ALL}: Seed {seed} ({Fore.BLUE}{Style.BRIGHT}{self._status.working_taxo.get_label(seed)}{Style.RESET_ALL})', 2, 'outer_loop')
                    outer_loop_progress, _ = self.outer_loop(seed)
                    self._status.progress += outer_loop_progress
                    if self._status.pbar_outer:
                        self._status.pbar_outer.update()

    def manual(self, **kwargs) -> None:

        self.update_config(**kwargs)
        if not self.config.manual_config.input_concepts:
            raise ValueError('Please provide a list of manual inputs in manual mode')
        if self.config.manual_config.auto_bases:
            if self.check_vector_index_available(self._status.working_taxo):
                self.print_log('Found pre-computed vector index', 1, 'system')
            else:
                self.print_log('Building vector index...', 1, 'system', newline=False)
                self.build_vector_index(self._status.working_taxo)
                self.print_log('Complete', 1, 'system')
            vstore = self._caches.vector_store[id(self._status.working_taxo)]
            _, inputs_bases = vstore.search(vstore.reconstruct(self.config.manual_config.input_concepts), k=self.config.ret_config.retrieve_size, exhaustive=True)
            inputs_bases = inputs_bases.tolist()
        elif not self.config.manual_config.manual_concept_bases:
            inputs_bases = [[]] * len(self.config.manual_config.input_concepts)
        elif len(self.config.manual_config.input_concepts) != len(self.config.manual_config.manual_concept_bases):
            raise ValueError('Lengths of input_concepts and manual_concept_bases must match')
        else:
            inputs_bases = self.config.manual_config.manual_concept_bases
        if self._status.pbar_outer:
            self._status.pbar_outer.reset(total = len(self.config.manual_config.input_concepts))
            self._status.pbar_outer.set_description('Manual mode')

        with self._status.pbar_outer:
            for i, newlabel in enumerate(self.config.manual_config.input_concepts):
                self.print_log(f'Input: {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL}', 2, 'outer_loop')
                self._status.outer_loop_count += 1
                base = inputs_bases[i]
                if base == []:
                    self.print_log('No search base available', 3, 'inner_loop')
                else:
                    plural_str = 'es' if len(base) > 1 else ''
                    self.print_log(f'Search will be based on the following class{plural_str}:', 3, 'inner_loop')
                    for b in base:
                        self.print_log(f'{Fore.BLUE}{Style.BRIGHT}{self._status.working_taxo.get_label(b)}{Style.RESET_ALL}', 4, 'outer_loop_concept_list')

                inner_loop_progress = self.inner_loop(newlabel, base)
                self._status.progress += inner_loop_progress
                if self._status.pbar_outer:
                    self._status.pbar_outer.update()

    def run(self, **kwargs) -> Union[Taxonomy, dict]:

        self.update_config(**kwargs)
        if self.config.rand_seed is not None:
            np.random.seed(self.config.rand_seed)
            torch.manual_seed(self.config.rand_seed)
        if not self.data:
            raise ValueError('Missing input data')
        self._status.working_taxo = deepcopy(self.data)
        if self.config.update_config.do_lexical_check:
            self.load_lexical_cache(self._status.working_taxo, self._caches.lexical_cache[id(self.data)])
        self.print_log(f'Loaded {self.data.__str__()}. Commencing enrichment', 1, 'system')

        self._status.outer_loop_count = 0
        self._status.progress = np.array([0,0], dtype=int)
        self._status.nextkey = max(hash(n) for n in self._status.working_taxo.nodes)+1
        progress_bar = (isinstance(self.config.logging,bool) and self.config.logging) or (isinstance(self.config.logging,int) and self.config.logging >= 1) or (isinstance(self.config.logging,list) and 'progress_bar' in self.config.logging)
        if progress_bar:
            self._status.pbar_outer = tqdm(total = 1, position = 0)
            self._status.pbar_inner = tqdm(total = 1, position = 1, leave=False) if self.config.mode in ['auto', 'semiauto'] else NullContext()
        else:
            self._status.pbar_outer = NullContext()
            self._status.pbar_inner = NullContext()

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
                raise ModuleNotFoundError('sub_model is required to run manual mode')
            elif self.config.manual_config.auto_bases and self.models.emb_model is None:
                raise ModuleNotFoundError('emb_model is required to run manual mode with auto_bases == True')
            self.manual()

        suffix = ' with transitive reduction' if self.config.transitive_reduction else ''
        plural_node = 's' if self._status.progress[0] > 1 else ''
        plural_edge = 's' if self._status.progress[1] > 1 else ''
        progress_str = f' Added {Fore.BLACK}{Style.BRIGHT}{self._status.progress[0]}{Style.RESET_ALL} new class{plural_node} and {Fore.BLACK}{Style.BRIGHT}{self._status.progress[1]}{Style.RESET_ALL} new direct subsumption{plural_edge}.' if self.config.update_config.do_update else ''
        self.print_log(f'Enrichment complete.{progress_str} Begin post-processing{suffix}', 1, 'system')
        if self.config.transitive_reduction:
            tr = nx.transitive_reduction(self._status.working_taxo)
            edgediff = set(self._status.working_taxo.edges).difference(tr.edges)
            self._status.working_taxo.remove_edges_from(edgediff)
        self._status.working_taxo.add_edges_from((u, v, self.data.edges[u, v]) for u, v in self.data.edges)

        if self.config.update_config.do_update:
            self.print_log(f'Return {self._status.working_taxo.__str__()}', 1, 'system')
            output = self._status.working_taxo
        else:
            self.print_log('Return ICON predictions', 1, 'system')
            output = self._status.logs
        self.clear_lexical_cache(self._status.working_taxo)
        self._status = _config.icon_status()

        return output
