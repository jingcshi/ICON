from typing import List, Union, Tuple, Iterable, Hashable, Callable, Dict, Any, Literal
from dataclasses import dataclass, field, fields, replace
from utils.taxo_utils import Taxonomy
import re
import numpy as np

@dataclass
class tree_config:
    
    def arglist(self, flat: bool = True):
        
        args = []
        for f in fields(self):
            F = getattr(self,f.name)
            if isinstance(F,tree_config):
                if flat:
                    args += F.arglist(flat = True)
                else:
                    args.append({f.name: F.arglist(flat = False)})
            else:
                args.append(f.name)
        return args
    
    def leaf_fields(self):
        
        return [f.name for f in fields(self) if not isinstance(getattr(self,f.name),tree_config)]
    
    def nonleaf_fields(self):
        
        return [f.name for f in fields(self) if isinstance(getattr(self,f.name),tree_config)]

@dataclass
class icon_models(tree_config):
    ret_model: Any = None
    gen_model: Any = None
    sub_model: Any = None

@dataclass
class icon_caches(tree_config):
    lexical_cache: Dict = field(default_factory = dict)
    sub_score_cache: Dict = field(default_factory = dict)

@dataclass
class icon_status(tree_config):
    nextkey: int = 0
    outer_loop_count: int = 0
    inner_loop_count: int = 0
    pbar_outer: Any = None
    pbar_inner: Any = None
    working_taxo: Taxonomy = None
    progress: np.ndarray = np.array([0,0], dtype = int)
    logs: Dict = field(default_factory = dict)

@dataclass
class icon_auto_config(tree_config):
    max_outer_loop: int = None

@dataclass
class icon_semiauto_config(tree_config):
    semiauto_seeds: List[Union[int, str]]=field(default_factory = list)

@dataclass
class icon_manual_config(tree_config):
    input_concepts: List[str]=field(default_factory = list)
    manual_concept_bases: List[List[Union[int, str]]] = None
    auto_bases: bool = False
    
@dataclass
class icon_ret_config(tree_config):
    retrieve_size: int = 10
    restrict_combinations: bool = True
    
@dataclass
class icon_gen_config(tree_config):
    ignore_label: List[str]=field(default_factory = list)
    filter_subset: bool = True

@dataclass
class icon_subgraph_config(tree_config):
    subgraph_crop: bool = True
    subgraph_force: List[List[str]] = field(default_factory = list)
    subgraph_strict: bool = True

@dataclass
class icon_search_config(tree_config):
    threshold: float = 0.5
    tolerance: int = 0
    force_base_subsumptions: bool = False
    force_prune: bool = False
    
@dataclass
class icon_sub_config(tree_config):
    subgraph: icon_subgraph_config = icon_subgraph_config()
    search: icon_search_config = icon_search_config()

@dataclass
class icon_update_config(tree_config):
    do_update: bool = True
    eqv_score_func: Callable[[Tuple[float, float]], float]=lambda x: x[0]*x[1]
    do_lexical_check: bool = True

@dataclass
class icon_config(tree_config):
    mode: Literal['auto', 'semiauto', 'manual']='auto'
    rand_seed: Any = 114514
    auto_config: icon_auto_config = icon_auto_config()
    semiauto_config: icon_semiauto_config = icon_semiauto_config()
    manual_config: icon_manual_config = icon_manual_config()
    ret_config: icon_ret_config = icon_ret_config()
    gen_config: icon_gen_config = icon_gen_config()
    sub_config: icon_sub_config = icon_sub_config()
    update_config: icon_update_config = icon_update_config()
    transitive_reduction: bool = True
    logging: Union[bool, int, List[str]]=1

@dataclass
class iconforcategorymove_ret_config(icon_ret_config):
    candidate_top_level: int = -1 # absolute levels
    candidate_bottom_level: int = 1
    ret_ignore: Union[Iterable[Hashable], re.Pattern] = field(default_factory = list)

@dataclass
class iconforcategorymove_subgraph_config(icon_subgraph_config):
    scope_top_level: int = 0
    scope_bottom_level: int = 1
    sub_ignore: Union[Iterable[Hashable], re.Pattern] = field(default_factory = list)

@dataclass
class iconforcategorymove_search_config(icon_search_config):
    always_search_to_bottom : bool = True

@dataclass
class iconforcategorymove_sub_config(icon_sub_config):
    subgraph: iconforcategorymove_subgraph_config = iconforcategorymove_subgraph_config()
    search: iconforcategorymove_search_config = iconforcategorymove_search_config()

@dataclass
class iconforcategorymove_selection_config(tree_config):
    do_select: bool = True
    always_include_old: bool = True
    selection_features: List[Literal['parent', 'siblings']] = field(default_factory = list)
    weights: np.ndarray = np.array([1,1], dtype = float)

@dataclass
class iconforcategorymove_auto_config(icon_auto_config):
    ignore: Union[Iterable[Hashable], re.Pattern] = field(default_factory = list)

@dataclass
class iconforcategorymove_manual_config(icon_manual_config):
    input_concepts: Iterable[Hashable]=field(default_factory = list)

@dataclass
class iconforcategorymove_config(icon_config):
    mode: Literal['auto', 'manual'] = 'auto'
    method: Literal['search', 'rag'] = 'search'
    ret_config: iconforcategorymove_ret_config = iconforcategorymove_ret_config()
    sub_config: iconforcategorymove_sub_config = iconforcategorymove_sub_config()
    selection_config: iconforcategorymove_selection_config = iconforcategorymove_selection_config()
    auto_config: iconforcategorymove_auto_config = iconforcategorymove_auto_config()
    manual_config: iconforcategorymove_manual_config = iconforcategorymove_manual_config()

def locate_arg(conf: tree_config, arg: str) -> str:

    if arg in conf.leaf_fields():
        return arg
    for f in conf.nonleaf_fields():
        subloc = locate_arg(getattr(conf, f), arg)
        if subloc:
            return f + '.' + subloc
    return ''

def Update_config(conf: tree_config, arg: str, value: Any) -> tree_config:
    
    location = locate_arg(conf,arg)
    if location == '':
        raise KeyError(f'{arg}')
    return recursive_replace(conf, location, value)

def recursive_replace(root_obj: Any, replace_str: str, replace_with: Any) -> Any:
    
    split_str = replace_str.split(".")
    if len(split_str) == 1:
        return replace(root_obj, **{split_str[0]: replace_with})
    sub_obj = recursive_replace(getattr(root_obj, split_str[0]), ".".join(split_str[1:]), replace_with)
    return replace(root_obj, **{split_str[0]: sub_obj})