import os
import sys
sys.path.append(os.getcwd() + '/..')
from typing import List, Union, Tuple, Callable, Dict, Any
from itertools import combinations
from collections import deque
import torch
import numpy as np
import owlready2 as o2
import networkx as nx
from tqdm.notebook import tqdm
from colorama import Fore, Style
from utils.taxo_utils import taxonomy
from utils.breadcrumb import tokenset
import utils.config as Config

class NullContext:
    
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass
    def __bool__(self):
        return False

def update_data(
    taxo:taxonomy,
    new:str,
    newkey:Union[int, str]=None,
    eqv:Union[int, str]=None,
    sup:List[Union[int, str]]=[],
    sub:List[Union[int, str]]=[],
    log:Union[bool, int, List[str]]=False):
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
    sup_set = set()
    sub_set = set()
    superr_set = set()
    suberr_set = set()
    
    if not eqv and not newkey:
        raise ValueError('Neither a new key nor an equivalent class key is specified')
    else:
        # Clean up the superclass / subclass sets. We only want to add the most specific superclasses and most general subclasses
        sup = taxo.reduce_subset(sup)
        sub = taxo.reduce_subset(sub,reverse=True)
        
    # Case 1: Merge with a known equivalent class    
    if eqv:
        if eqv in taxo.nodes:
            print_log(f'Declared {Fore.MAGENTA}{Style.BRIGHT}equivalence{Style.RESET_ALL} between {Fore.YELLOW}{Style.BRIGHT}{taxo.get_label(eqv)}{Style.RESET_ALL} ({eqv}) and {Fore.CYAN}{Style.BRIGHT}{new}{Style.RESET_ALL}', log, 4, 'iter_details')
            selfclass = eqv
            self_color = Fore.YELLOW
            self_label = taxo.get_label(eqv)
        else:
            raise KeyError(f'Equivalent class {newkey} not found')
        
    # Case 2: Add a new class
    else:
        if taxo.add_node(newkey,label=new) == 0:
            print_log(f'{Fore.GREEN}{Style.BRIGHT}Created{Style.RESET_ALL} new class {Fore.CYAN}{Style.BRIGHT}{new}{Style.RESET_ALL} with key {Fore.BLACK}{Style.BRIGHT}{newkey}{Style.RESET_ALL}', log, 4, 'iter_details')
            selfclass = newkey
            self_color = Fore.CYAN
            self_label = new
        else:
            raise KeyError(f'Key conflict: {newkey}')
    
    # Update taxonomies
    for superclass in sup:
        try:
            if taxo.add_edge(selfclass,superclass,label='added') == 0:
                sup_set.add(superclass)
        except TypeError:
            superr_set.add(superclass)
    for subclass in sub:
        try:
            if taxo.add_edge(subclass,selfclass,label='added') == 0:
                sub_set.add(subclass)
        except TypeError:
            suberr_set.add(subclass)

    # Log actions
    if sup_set:
        verb = 'are' if len(sup_set) > 1 else 'is'
        suffix = 'es' if len(sup_set) > 1 else ''
        print_log(f'The following class{suffix} {verb} declared as {Fore.MAGENTA}{Style.BRIGHT}superclass{suffix}{Style.RESET_ALL} of {self_color}{Style.BRIGHT}{self_label}{Style.RESET_ALL}:', log, 5, 'iter_classlist')
        for supc in sup_set:
            print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(supc)} {Style.RESET_ALL}({supc})', log, 5, 'iter_classlist')
    if sub_set:
        verb = 'are' if len(sub_set) > 1 else 'is'
        suffix = 'es' if len(sub_set) > 1 else ''
        print_log(f'The following class{suffix} {verb} declared as {Fore.MAGENTA}{Style.BRIGHT}subclass{suffix}{Style.RESET_ALL} of {self_color}{Style.BRIGHT}{self_label}{Style.RESET_ALL}:', log, 5, 'iter_classlist')
        for subc in sub_set:
            print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(subc)} {Style.RESET_ALL}({subc})', log, 5, 'iter_classlist')

    err_set = superr_set.union(suberr_set)
    if err_set:
        verb = 'are' if len(err_set) > 1 else 'is'
        suffix = 's' if len(err_set) > 1 else ''
        print_log(f'{Fore.RED}{Style.BRIGHT}Warning{Style.RESET_ALL}: The following subClassOf relation{suffix} {verb} {Fore.BLACK}{Style.BRIGHT}discarded{Style.RESET_ALL} because of cyclic inheritance:', log, 5, 'iter_classlist')
        for supc in superr_set:
            print_log(f'\t{self_color}{Style.BRIGHT}{self_label} {Fore.BLACK}--> {Fore.BLUE}{taxo.get_label(supc)} {Style.RESET_ALL}({supc})', log, 5, 'iter_classlist')
        for subc in suberr_set:
                print_log(f'\t{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(subc)} {Style.RESET_ALL}({subc}) {Fore.BLACK}{Style.BRIGHT}--> {self_color}{self_label}{Style.RESET_ALL}', log, 5, 'iter_classlist')
    
    return np.array([int(not eqv),len(sup_set)+len(sub_set)], dtype=int)

def enhanced_traversal(
                    taxo:taxonomy,
                    query:str,
                    model,
                    cached_subs_scores:Dict={},
                    base:List[Union[int, str]]=None,
                    threshold:float=0.5,
                    tolerance:int=0,
                    force_known:bool=False,
                    force_prune:bool=False):
    '''
    Search for the optimal placement (equivalences, superclasses and subclasses) of a new concept (represented by text) in a given taxonomy
    The basic algorithm is a two-stage BFS, one top-down in search for superclasses and one bottom-up in search for subclasses

    Args:
        taxo: The search domain. Does not have to be the full taxonomy
        query: Concept to insert
        model: Subsumption prediction model. Should accept input params in the shape of (subclass:str,superclass:str)
        base: The seed classes (represented by keys) used to generate the query. Has effect only if force_known=True
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
    if threshold > 1 or threshold < 0:
        raise ValueError('Threshold must be in the range [0,1]')
    
    force_known = force_known and base
    # Stage 1: search for superclasses
    sup = {}
    # If force_known=True, the starting point of this search becomes the LCA of the base w.r.t. the original taxonomy, plus any other LCA. Otherwise, the starting point is get_GCD(empty set) which returns the top nodes in the domain
    if force_known:
        top = taxo.get_LCA(base,return_type=set).union(taxo.get_LCA(base,labels='original',return_type=set))
        top = taxo.reduce_subset(top,reverse=True)
    else:
        top = taxo.get_GCD([])
    queue = deque([(n,0) for n in top])
    if top:
        cache_score(cached_subs_scores,([taxo.get_label(n) for n in top],[query]*len(top)),model)
    visited = {}

    while queue:
        node, fails = queue.popleft()
        visited[node] = True
        to_cache = []
        keypair = (query,taxo.get_label(node))
        if keypair in cached_subs_scores:
            p = cached_subs_scores[keypair]
        # Key zero is always assumed to be the global root node
        elif node == 0:
            p = 1.0
        else:
            p = model(keypair).item()
            cached_subs_scores[keypair] = p
        
        if p >= threshold:
            sup[node] = p
            if force_known:
                if node in base:
                    # In the superclass search stage, the base classes mark the end of search on the current branch (the query should never be more specific than the base). It is possible, however, that some of the seeds subsume the query.
                    if force_prune:
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
                cache_score(cached_subs_scores,([query]*len(to_cache),to_cache),model)
        elif fails < tolerance:
            for child in taxo.get_subclasses(node):
                if child not in visited:
                    # Keep searching down until success or failures accumulate to tolerance. Used to alleviate misjudgments of the model
                    queue.append((child,fails+1))
                    to_cache.append(taxo.get_label(child))
            if to_cache:
                cache_score(cached_subs_scores,([query]*len(to_cache),to_cache),model)
        elif force_prune:
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
        cache_score(cached_subs_scores,([taxo.get_label(n) for n in bottom],[query]*len(bottom)),model)
    # The redundant superclasses are also logically certain to be non-subclasses, so we do not search on them
    visited = {k:True for k in sup_ancestors}
    
    while queue:
        node, fails = queue.popleft()
        visited[node] = True
        to_cache = []
        keypair = (taxo.get_label(node),query)
        if keypair in cached_subs_scores:
                p = cached_subs_scores[keypair]
        elif force_known:
            if node in base:
                # Force the search to respect known subsumptions
                p = 1.0
            else:
                p = model(keypair).item()
        else:
            p = model(keypair).item()
            cached_subs_scores[keypair] = p
            
        if p >= threshold:
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
                cache_score(cached_subs_scores,(to_cache,[query]*len(to_cache)),model)
        elif fails < tolerance:
            # Keep searching up until success or failures accumulate to tolerance
            for parent in taxo.get_superclasses(node):
                if parent not in visited:
                    queue.append((parent,fails+1))
                    to_cache.append(taxo.get_label(parent))
            if to_cache:
                cache_score(cached_subs_scores,(to_cache,[query]*len(to_cache)),model)
        elif force_prune:
            for ance in taxo.get_ancestors(node):
                visited[ance] = True
    
    # Same story as in stage 1, we only keep the maximal subclasses
    if sub:
        sub = {k:sub[k] for k in taxo.reduce_subset(list(sub.keys()),reverse=True)}
    
    return sup, sub, eqv    

def generate(
        taxo:taxonomy,
        base:List[Union[int, str]],
        gen_model,
        ignore_label:List[str]=['','All categories','All products','Thing','Allcats','Everything','root'],
        log:Union[bool, int, List[str]]=False):
    
    # Skip if a direct (one level or less above) CP in the original taxonomy exists, or if the LCA is already in base
    has_direct_cp = bool(set.intersection(*[taxo.get_superclasses(c,labels='original',return_type=set) for c in base]))
    if any([has_direct_cp, taxo.get_LCA(base,return_type=set).issubset(set(base))]):
        print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Skipped{Style.RESET_ALL} because a direct common parent exists', log, 3, 'iter')
        return None

    # Generate new CP label
    newlabel = gen_model([taxo.get_label(b) for b in base])

    # Reject if the generated label is considered root
    if newlabel in ignore_label:
        print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Rejected{Style.RESET_ALL} by label generator', log, 3, 'iter')
        return None

    print_log(f'Generated common parent label: {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL}', log, 3, 'iter')
    return newlabel

def run_iteration(
        taxo:taxonomy,
        newlabel:str,
        sub_model,
        nextkey:Union[int, str],
        cached_subs_scores:Dict={},
        base:List[Union[int, str]]=[],
        subgraph_crop:bool=True,
        subgraph_force:List[List[str]]=[['original']],
        subgraph_strict:bool=True,
        subs_threshold:float=0.5,
        search_tolerance:int=0,
        force_known_subsumptions:bool=False,
        force_prune_branches:bool=False,
        eqv_score_func:Callable[[Tuple[float, float]], float]=lambda x: x[0]*x[1],
        nil_cache:Dict[int, Union[int, str]]={},
        log:Union[bool, int, List[str]]=False):
    
    iter_results = {'equivalence': None, 'superclass': None, 'subclass': None}
    
    # Create search domain.
    subtaxo = taxo.create_subgraph(base,crop_top=subgraph_crop,force_labels=subgraph_force,strict=subgraph_strict)
    subgraph_size = len(subtaxo.nodes)
    print_log(f'Searching on a domain of {subgraph_size} classes', log, 4, 'iter_details')

    # Search for optimal placement using the SUB model to predict subsumption.
    sup, sub, eqv = enhanced_traversal(subtaxo,newlabel,model=sub_model,cached_subs_scores=cached_subs_scores,base=base,threshold=subs_threshold,tolerance=search_tolerance,force_known=force_known_subsumptions,force_prune=force_prune_branches)

    resolution = lexical_check(nil_cache,newlabel)
    if resolution:
        # We give 100% confidence to the linkage of NIL model, thus making it supercede any other possible equivalences provided by search
        eqv[resolution] = (1.0,1.0)
        print_log(f'\tSearch complete. {Fore.YELLOW}{Style.BRIGHT}Mapped{Style.RESET_ALL} to a known class by NIL entity resolver', log, 3, 'iter')
    else:
        print_log(f'\tSearch complete. {Fore.GREEN}{Style.BRIGHT}Validated{Style.RESET_ALL} by NIL entity resolver', log, 3, 'iter')

    # Reject the new class because there is no good placement
    if not sup and not eqv:
        print_log(f'\t{Fore.BLACK}{Style.BRIGHT}Rejected{Style.RESET_ALL} by search because no good placement can be found', log, 3, 'iter')
        return np.array([0,0], dtype=int), iter_results
    # When there are more than one equivalent classes, keep only the most confident equivalent class
    # Demote the other equivalent classes to either superclasses or subclasses, whichever got the higher likelihood
    if len(eqv) > 1:
        ranked_eqvclasses = [k for k, v in sorted(eqv.items(), key=lambda x: eqv_score_func(x[1]), reverse=True)]
        do_nil_string = ' / NIL entity resolution' if resolution else ''
        print_log(f'{Fore.RED}{Style.BRIGHT}Warning{Style.RESET_ALL}: Search{do_nil_string} suggests that {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL} is {Fore.MAGENTA}{Style.BRIGHT}equivalent{Style.RESET_ALL} to multiple known classes', log, 4, 'iter_details')
        for eqvclass in ranked_eqvclasses:
            classlabel = taxo.get_label(eqvclass)
            prob = eqv_score_func(eqv[eqvclass])
            print_log(f'{Fore.BLUE}{Style.BRIGHT}{classlabel}{Style.RESET_ALL} with score {prob:.4f}', log, 5, 'iter_classlist')
        for eqvclass in ranked_eqvclasses[1:]:
            if eqv[eqvclass][0] >= eqv[eqvclass][1]:
                sup[eqvclass] = eqv.pop(eqvclass)[0]
            else:
                sub[eqvclass] = eqv.pop(eqvclass)[1]
        print_log(f'For safety, only the highest ranked equivalence is preserved', log, 4, 'iter_details')

    if eqv:
        eqvc = list(eqv)[0]
        sup.pop(eqvc, None)
        sub.pop(eqvc, None)
        iter_results = {'equivalence': eqv, 'superclass': sup, 'subclass': sub}
        eqv = eqvc
        if not resolution:
            print_log(f'\t{Fore.YELLOW}{Style.BRIGHT}Mapped{Style.RESET_ALL} to a known class by search', log, 3, 'iter')
    else:
        iter_results = {'equivalence': eqv, 'superclass': sup, 'subclass': sub}
        print_log(f'\t{Fore.GREEN}{Style.BRIGHT}Accepted{Style.RESET_ALL} as a new class by search', log, 3, 'iter')
    
    iter_progress = update_data(taxo,newlabel,newkey=nextkey,eqv=eqv,sup=list(sup),sub=list(sub),log=log)
    if iter_progress[0].item():
        nil_cache[hash(tuple(tokenset(newlabel)))] = nextkey
    return iter_progress, iter_results

def run_cycle(
        taxo:taxonomy,
        knn_model,
        gen_model,
        sub_model,
        seed:Union[int, str],
        nextkey:Union[int, str],
        retrieve_size:int=10,
        restrict_combinations=True,
        ignore_label:List[str]=['','All categories','All products','Thing','Allcats','Everything','root'],
        cached_subs_scores:Dict={},
        subgraph_crop:bool=True,
        subgraph_force:List[List[str]]=[['original']],
        subgraph_strict:bool=True,
        subs_threshold:float=0.5,
        search_tolerance:int=0,
        force_known_subsumptions:bool=False,
        force_prune_branches:bool=False,
        eqv_score_func:Callable[[Tuple[float, float]], float]=lambda x: x[0]*x[1],
        nil_cache:Dict[int, Union[int, str]]={},
        log:Union[bool, int, List[str]]=False,
        cycle_count:int=None,
        pbar=None):
    
    cycle_progress = np.array([0,0],dtype=int)
    
    # Retrieve a set of relevant classes from the KNN model, henceforce referred to as base_classes
    base_classes = knn_model(taxo,seed,k=retrieve_size)
    print_log(f'Retrieved {Fore.BLACK}{Style.BRIGHT}{len(base_classes)}{Style.RESET_ALL} classes', log, 3, 'cycle_details')
    for c in base_classes:
        print_log(f'{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(c)}{Style.RESET_ALL}', log, 4, 'cycle_classlist')
    
    # Use base class pairs as prompt for the GEN model to generate new class names
    if restrict_combinations:
        non_seed = base_classes.copy()
        non_seed.remove(seed)
        iter_inputs = [(seed, b) for b in non_seed]
    else:
        iter_inputs = list(combinations(base_classes, 2))
    
    if pbar:
        pbar.reset(total = len(iter_inputs))
        if cycle_count:
            pbar.set_description(f'Cycle {cycle_count}')
    for i,subset in enumerate(iter_inputs):
        prefix = f'{cycle_count}.' if cycle_count else ''
        msg = f'Iteration {Fore.BLACK}{Style.BRIGHT}{prefix}{i+1}{Style.RESET_ALL}: Combination ('
        for b in subset:
            msg += f'{Fore.BLUE}{Style.BRIGHT}{taxo.get_label(b)}{Style.RESET_ALL}, '
        print_log(msg[:-2] + ')', log, 3, 'iter')
        
        newlabel = generate(taxo=taxo,base=subset,gen_model=gen_model,ignore_label=ignore_label,log=log)
        
        if newlabel:
            iter_progress, iter_results = run_iteration(taxo=taxo,newlabel=newlabel,sub_model=sub_model,nextkey=nextkey,cached_subs_scores=cached_subs_scores,base=list(subset),ignore_label=ignore_label,subgraph_crop=subgraph_crop,subgraph_force=subgraph_force,subgraph_strict=subgraph_strict,subs_threshold=subs_threshold,search_tolerance=search_tolerance,force_known_subsumptions=force_known_subsumptions,force_prune_branches=force_prune_branches,eqv_score_func=eqv_score_func,nil_cache=nil_cache,log=log)
            cycle_progress += iter_progress
            nextkey += iter_progress[0].item()
        if pbar:
            pbar.update()
        
    print_log(f'Cycle complete. Added {Fore.BLACK}{Style.BRIGHT}{cycle_progress[0]}{Style.RESET_ALL} new classes and {Fore.BLACK}{Style.BRIGHT}{cycle_progress[1]}{Style.RESET_ALL} new direct subsumptions.', log, 2, 'cycle')
    return cycle_progress, set(base_classes)

def main(data:Union[taxonomy, o2.Ontology],
        knn_model=None,
        gen_model=None,
        sub_model=None,
        mode:str='auto',
        max_cycle:int=None,
        semiauto_seeds:List[Union[int, str]]=[],
        manual_inputs:List[str]=[],
        inputs_bases:List[List[Union[int, str]]]=None,
        rand_seed=20230103,
        retrieve_size:int=10,
        restrict_combinations=True,
        ignore_label:List[str]=['','All categories','Root Concept','Thing','Allcats','Everything','root'],
        cached_subs_scores:Dict={},
        subgraph_crop:bool=True,
        subgraph_force:List[List[str]]=[['original']],
        subgraph_strict:bool=True,
        subs_threshold:float=0.5,
        search_tolerance:int=0,
        force_known_subsumptions:bool=False,
        force_prune_branches:bool=False,
        eqv_score_func:Callable[[Tuple[float, float]], float]=lambda x: x[0]*x[1],
        transitive_reduction:bool=True,
        log:Union[bool, int, List[str]]=False,
        ):
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
                max_cycle: Maximal number of cycles allowed. Used for auto mode.
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

    if rand_seed != None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
    # Extract taxonomy from ontology, or use the given taxonomy if provided
    if isinstance(data,o2.Ontology):
        data = taxonomy.from_ontology(data)
    taxo = data
    nil_cache = {}
    with tqdm(total = len(taxo), leave=False, desc='Loading') as pbar:
        for n in taxo.nodes:
            # The dictionary used for NIL entity check. Keys are hash values of tokenised and lemmatised class labels
            nil_cache[hash(tuple(tokenset(taxo.get_label(n))))] = n
            pbar.update(1)
    
    print_log(f'Loaded {taxo.__str__()}. Commencing enrichment', log, 1, 'system')
    nextkey = max(hash(n) for n in taxo.nodes)+1 # Track the next ID in case of new class insertion
    progress = np.array([0,0],dtype=int)
    cycle_count = 0 # Track progress in the form of [new classes, new direct subsumptions]
    progress_bar = (isinstance(log,bool) and log) or (isinstance(log,int) and log >= 1) or (isinstance(log,list) and 'progress_bar' in log)
    if not progress_bar:
        pbar_outer = NullContext()
        pbar_inner = NullContext()
    
    # Sample a random untouched bottom class as seed each cycle, or use the given seeds if provided
    if mode == 'auto':
        seedpool = taxo.get_LCA([],return_type=set)
        poolsize = len(seedpool)
        if not max_cycle:
            max_cycle = poolsize
        if progress_bar:
            pbar_outer = tqdm(total = poolsize, position = 0, desc='Auto mode')
            pbar_inner = tqdm(total = 1, position = 1, leave=False)
            
        with pbar_outer:
            with pbar_inner:
                while cycle_count < max_cycle and seedpool:
                    candidates = list(seedpool)
                    poolsize = len(candidates)
                    seed = candidates[np.random.choice(poolsize,1).item()]
                    cycle_count += 1
                    print_log(f'Cycle {Fore.BLACK}{Style.BRIGHT}{cycle_count}{Style.RESET_ALL}: Seed {seed} ({Fore.BLUE}{Style.BRIGHT}{taxo.get_label(seed)}{Style.RESET_ALL}) selected from {Fore.BLACK}{Style.BRIGHT}{poolsize}{Style.RESET_ALL} possible candidates', log, 2, 'cycle')
                    cycle_progress, processed = run_cycle(taxo=taxo,knn_model=knn_model,gen_model=gen_model,sub_model=sub_model,seed=seed,cached_subs_scores=cached_subs_scores,nextkey=nextkey,retrieve_size=retrieve_size,restrict_combinations=restrict_combinations,ignore_label=ignore_label,subgraph_crop=subgraph_crop,subgraph_force=subgraph_force,subgraph_strict=subgraph_strict,subs_threshold=subs_threshold,search_tolerance=search_tolerance,force_known_subsumptions=force_known_subsumptions,force_prune_branches=force_prune_branches,eqv_score_func=eqv_score_func,nil_cache=nil_cache,log=log,cycle_count=cycle_count,pbar=pbar_inner)
                    progress += cycle_progress
                    nextkey += cycle_progress[0].item()
                    seedpool = seedpool.difference(processed)
                    if progress_bar:
                        new_poolsize = len(seedpool)
                        diff = poolsize - new_poolsize
                        poolsize = new_poolsize
                        pbar_outer.update(diff)
                        
    elif mode == 'semiauto':
        if not semiauto_seeds:
            raise ValueError('Please provide a list of seeds in semiauto mode')
        if progress_bar:
            pbar_outer = tqdm(total = len(semiauto_seeds), position = 0, desc='Semiauto mode')
            pbar_inner = tqdm(total = 1, position = 1, leave=False)
            
        with pbar_outer:
            with pbar_inner:
                for seed in semiauto_seeds:
                    cycle_count += 1
                    print_log(f'Cycle {Fore.BLACK}{Style.BRIGHT}{cycle_count}{Style.RESET_ALL}: Seed {seed} ({Fore.BLUE}{Style.BRIGHT}{taxo.get_label(seed)}{Style.RESET_ALL})', log, 2, 'cycle')
                    cycle_progress, processed = run_cycle(taxo=taxo,knn_model=knn_model,gen_model=gen_model,sub_model=sub_model,seed=seed,nextkey=nextkey,cached_subs_scores=cached_subs_scores,retrieve_size=retrieve_size,restrict_combinations=restrict_combinations,ignore_label=ignore_label,subgraph_crop=subgraph_crop,subgraph_force=subgraph_force,subgraph_strict=subgraph_strict,subs_threshold=subs_threshold,search_tolerance=search_tolerance,force_known_subsumptions=force_known_subsumptions,force_prune_branches=force_prune_branches,eqv_score_func=eqv_score_func,nil_cache=nil_cache,log=log,cycle_count=cycle_count,pbar=pbar_inner)
                    progress += cycle_progress
                    nextkey += cycle_progress[0].item()
                    if progress_bar:
                        pbar_outer.update()
                        
    elif mode == 'manual':
        if not manual_inputs:
            raise ValueError('Please provide a list of manual inputs in manual mode')
        if not inputs_bases:
            inputs_bases = [[]] * len(manual_inputs)
        if len(manual_inputs) != len(inputs_bases):
            raise ValueError('Lengths of manual_inputs and inputs_bases must match')
        if progress_bar:
            pbar_outer = tqdm(total = len(manual_inputs), desc='Manual mode')    
        results = {}
        
        with pbar_outer:
            for i,newlabel in enumerate(manual_inputs):
                print_log(f'Input: {Fore.CYAN}{Style.BRIGHT}{newlabel}{Style.RESET_ALL}', log, 3, 'iter')
                iter_progress, iter_results = run_iteration(taxo=taxo,newlabel=newlabel,sub_model=sub_model,nextkey=nextkey,cached_subs_scores=cached_subs_scores,base=inputs_bases[i],ignore_label=ignore_label,subgraph_crop=subgraph_crop,subgraph_force=subgraph_force,subgraph_strict=subgraph_strict,subs_threshold=subs_threshold,search_tolerance=search_tolerance,force_known_subsumptions=force_known_subsumptions,force_prune_branches=force_prune_branches,eqv_score_func=eqv_score_func,nil_cache=nil_cache,log=log)
                results[newlabel] = iter_results
                progress += iter_progress
                nextkey += iter_progress[0].item()
                if progress_bar:
                    pbar_outer.update()
    else:
        raise ValueError('Please select one of the following modes: auto, semiauto, manual')
    
    suffix = ' with transitive reduction' if transitive_reduction else ''
    print_log(f'Enrichment complete. Added {Fore.BLACK}{Style.BRIGHT}{progress[0]}{Style.RESET_ALL} new classes and {Fore.BLACK}{Style.BRIGHT}{progress[1]}{Style.RESET_ALL} new direct subsumptions. Begin post-processing{suffix}', log, 1, 'system')
    if transitive_reduction:
        tr = nx.transitive_reduction(taxo)
        tr.add_nodes_from(taxo.nodes(data=True))
        tr.add_edges_from((u, v, taxo.edges[u, v]) for u, v in tr.edges)
        taxo = taxonomy(tr)
    taxo.add_edges_from((u, v, data.edges[u, v]) for u, v in data.edges)
    
    print_log(f'Return {taxo.__str__()}', log, 1, 'system')
    
    if mode == 'manual':
        return taxo, results
    return taxo

class ICON:
    
    def __init__(self,
                data: Union[taxonomy,o2.Ontology]=None,
                ret_model=None,
                gen_model=None,
                sub_model=None,
                lexical_cache:Dict={},
                sub_score_cache:Dict={},
                mode:str='auto',
                max_outer_loop:int=None,
                semiauto_seeds:List[Union[int, str]]=[],
                input_concepts:List[str]=[],
                inputs_concept_bases:List[List[Union[int, str]]]=None,
                rand_seed:Any=114514,
                retrieve_size:int=10,
                restrict_combinations=True,
                ignore_label:List[str]=['','All categories','Root Concept','Thing','Allcats','Everything','root'],
                filter_subset:bool=True,
                subgraph_crop:bool=True,
                subgraph_force:List[List[str]]=[['original']],
                subgraph_strict:bool=True,
                threshold:float=0.5,
                tolerance:int=0,
                force_base_subsumptions:bool=False,
                force_prune:bool=False,
                eqv_score_func:Callable[[Tuple[float, float]], float]=lambda x: x[0]*x[1],
                do_lexical_check:bool=True,
                transitive_reduction:bool=True,
                log:Union[bool, int, List[str]]=False,
                ):
        
        if isinstance(data,o2.Ontology):
            data = taxonomy.from_ontology(data)
        self.data = data
        self.models = Config.icon_models(ret_model,gen_model,sub_model)
        self.caches = Config.icon_caches(lexical_cache,sub_score_cache)
        self.config = Config.icon_config(mode,
                                rand_seed,
                                Config.icon_auto_config(max_outer_loop),
                                Config.icon_semiauto_config(semiauto_seeds),
                                Config.icon_manual_config(input_concepts,inputs_concept_bases),
                                Config.icon_ret_config(retrieve_size,restrict_combinations),
                                Config.icon_gen_config(ignore_label,filter_subset),
                                Config.icon_sub_config(
                                    Config.icon_subgraph_config(subgraph_crop,subgraph_force,subgraph_strict),
                                    Config.icon_search_config(threshold,tolerance,force_base_subsumptions,force_prune)),
                                Config.icon_update_config(eqv_score_func,do_lexical_check),
                                transitive_reduction,
                                log)
        
        if data != None and do_lexical_check:
            self.load_lexical_cache(data)
        
    def load_lexical_cache(self,data:taxonomy):
        with tqdm(total = data.number_of_nodes(), leave=False, desc='Loading lexical cache') as pbar:
            for n,l in data.nodes(data='label'):
                # The dictionary used for NIL entity check. Keys are hash values of tokenised and lemmatised class labels
                self.caches.lexical_cache[hash(tuple(tokenset(l)))] = n
                pbar.update()
    
    def update_lexical_cache(self,node:Union[int,str],label:str):
        self.caches.lexical_cache[hash(tuple(tokenset(label)))] = node
    
    def clear_lexical_cache(self):
        self.caches.lexical_cache = {}
    
    def lexical_check(self,label):
        try:
            return self.caches.lexical_cache[hash(tuple(tokenset(label)))]
        except KeyError:
            return None
    
    def update_sub_score_cache(self,inputs:Tuple[List[str], List[str]]):
        subclasses, superclasses = inputs
        # The model output is expected to be a numpy.array of shape [len(keypair)]
        outputs = self.models.sub_model(inputs)
        for i,s in enumerate(outputs):
            self.caches.sub_score_cache[(subclasses[i],superclasses[i])] = s.item()
    
    def print_log(self, msg, level:int, msgtype:str):
        setting = self.config.log
        if (isinstance(setting,bool) and setting == True) or (isinstance(setting,int) and setting >= level) or (isinstance(setting,list) and msgtype in setting):
            indent = max(level-1, 0)
            print('\t' * indent + msg)
            return 1
        return 0

    def update_config(self,**kwargs):
        for arg, value in kwargs.items():
            Config.Update_config(self.config, arg, value)
