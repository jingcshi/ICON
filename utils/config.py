from typing import List, Union, Tuple, Callable, Dict, Any
from dataclasses import dataclass, field, fields, replace

@dataclass
class tree_config:
    
    def arglist(self, flat:bool=True):
        args = []
        for f in fields(self):
            F = getattr(self,f.name)
            if isinstance(F,tree_config):
                if flat:
                    args += F.arglist(flat=True)
                else:
                    args.append({f.name: F.arglist(flat=False)})
            else:
                args.append(f.name)
        return args
    
    def leaf_fields(self):
        return [f.name for f in fields(self) if not isinstance(getattr(self,f.name),tree_config)]
    
    def nonleaf_fields(self):
        return [f.name for f in fields(self) if isinstance(getattr(self,f.name),tree_config)]

@dataclass
class icon_models:
    ret_model:Any
    gen_model:Any
    sub_model:Any

@dataclass
class icon_caches:
    lexical_cache:Dict=field(default_factory=dict)
    sub_score_cache:Dict=field(default_factory=dict)

@dataclass
class icon_auto_config(tree_config):
    max_outer_loop:int=None

@dataclass
class icon_semiauto_config(tree_config):
    semiauto_seeds:List[Union[int, str]]=field(default_factory=list)

@dataclass
class icon_manual_config(tree_config):
    input_concepts:List[str]=field(default_factory=list)
    inputs_concept_bases:List[List[Union[int, str]]]=None

@dataclass
class icon_ret_config(tree_config):
    retrieve_size:int=10
    restrict_combinations:bool=True
    
@dataclass
class icon_gen_config(tree_config):
    ignore_label:List[str]=field(default_factory=list)
    filter_subset:bool=True

@dataclass
class icon_subgraph_config(tree_config):
    subgraph_crop:bool=True
    subgraph_force:List[List[str]]=field(default_factory=list)
    subgraph_strict:bool=True

@dataclass
class icon_search_config(tree_config):
    threshold:float=0.5
    tolerance:int=0
    force_base_subsumptions:bool=False
    force_prune:bool=False
    
@dataclass
class icon_sub_config(tree_config):
    subgraph:icon_subgraph_config=icon_subgraph_config()
    search:icon_search_config=icon_search_config()

@dataclass
class icon_update_config(tree_config):
    eqv_score_func:Callable[[Tuple[float, float]], float]=lambda x: x[0]*x[1]
    do_lexical_check:bool=True

@dataclass
class icon_config(tree_config):
    mode:str='auto'
    rand_seed:Any=114514
    auto_config:icon_auto_config=icon_auto_config()
    semiauto_config:icon_semiauto_config=icon_semiauto_config()
    manual_config:icon_manual_config=icon_manual_config()
    ret_config:icon_ret_config=icon_ret_config()
    gen_config:icon_gen_config=icon_gen_config()
    sub_config:icon_sub_config=icon_sub_config()
    update_config:icon_update_config=icon_update_config()
    transitive_reduction:bool=True
    log:Union[bool, int, List[str]]=False

def Update_config(conf:tree_config, arg:str, value:Any):
    if arg in conf.arglist(flat=True):
        if arg in conf.leaf_fields():
            conf = replace(conf, **{arg: value})
            return True
        else:
            for f in conf.nonleaf_fields():
                if Update_config(getattr(conf, f), arg, value):
                    break
            return True
    else:
        return False
        