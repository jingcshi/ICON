from __future__ import annotations
import re
import os
from typing import List, Set, Union, Literal, Type, Iterable, Hashable, Optional
from collections import deque
from copy import deepcopy
import owlready2 as o2
import networkx as nx
import json
import numpy as np

reID = re.compile(r'\d+')
reIRI = re.compile(r'#(.*)$')

class Taxonomy(nx.DiGraph):
    '''
    Class for taxonomy as used in ICON.
    
    A Taxonomy object is a directed acyclic graph (netwokx.DiGraph), with additional methods implemented to facilitate taxonomic operations.

        One special attribute 'label' should be used exclusively for the surface forms of concepts, as downstream method depend on it.
        
        The node key 0 is reserved for the root node and should not be re-defined for other nodes.
        
    All edges (u,v) are assumed to be subClassOf relations, meaning u is a subclass / hyponym of v.
    '''
    def __init__(self, *args, **kw) -> None:
        '''
        Define a Taxonomy object.
        
        Usage of this method is the same as netwokx.DiGraph.__init__, as no additional arguments are being processed.
        '''
        super().__init__(*args, **kw)
    
    def add_node(self, node_for_adding: Hashable, **attr) -> Literal[0,1,2]:
        '''
        Add a node to the taxonomy. Similar to networkx.DiGraph.add_node but adding return values.
        
        Interpretation of return values:
        
            0 if the new node is successfully inserted.
            
            1 if no action could be done.
            
            2 if the node already exists, but the attributes are updated.
        '''
        if node_for_adding not in self._succ:
            if node_for_adding is None:
                raise ValueError("None cannot be a node")
            self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
            self._pred[node_for_adding] = self.adjlist_inner_dict_factory()
            attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            attr_dict.update(attr)
            return 0
        elif attr:  # update attr even if node already exists
            self._node[node_for_adding].update(attr)
            return 2
        else:
            return 1
    
    def add_edge(self, u_of_edge: Hashable, v_of_edge: Hashable, **attr) -> Literal[0,1]:
        '''
        Add an edge (u, v) to the taxonomy. Similar to networkx.DiGraph.add_edge but adding return values.
        
        Interpretation of return values:
        
            0 if the new edge is successfully inserted.
            
            1 if the edge already exists, in which case the attributes are updated.
            
        Raises ValueError if the at least one of the relevant nodes is None.
        
        Raises TypeError if the edge to be added would cause a cycle, which would lead to a logical contradiction from a taxonomy's perspective.
        '''
        u, v = u_of_edge, v_of_edge
        # add nodes
        if u not in self._succ:
            if u is None:
                raise ValueError("None cannot be a node")
            self._succ[u] = self.adjlist_inner_dict_factory()
            self._pred[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._succ:
            if v is None:
                raise ValueError("None cannot be a node")
            self._succ[v] = self.adjlist_inner_dict_factory()
            self._pred[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        # add the edge
        if self.subsumes(u, v):
            raise TypeError('Edge not added because it would cause a cycle')
        return_value = 1 if v in self._succ[u] else 0 
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._succ[u][v] = datadict
        self._pred[v][u] = datadict
        return return_value
    
    def remove_node(self, n: Hashable) -> None:        
        '''
        Delete a node from the taxonomy. Similar to networkx.DiGraph.remove_node.
        
        Deleting a node will also remove all the edges attached to it.
        '''        
        try:
            nbrs = self._succ[n]
            del self._node[n]
        except KeyError as err:  # nx.NetworkXError if n not in self
            raise nx.NetworkXError(f"The node {n} is not in the digraph.") from err
        for u in nbrs:
            del self._pred[u][n]  # remove all edges n-u in digraph
        del self._succ[n]  # remove node from succ
        for u in self._pred[n]:
            del self._succ[u][n]  # remove all edges n-u in digraph
        del self._pred[n]  # remove node from pred
        
    def remove_edge(self, u: Hashable, v: Hashable) -> None:
        '''
        Delete an edge from the taxonomy. Similar to networkx.DiGraph.remove_edge.
        '''     
        try:
            del self._succ[u][v]
            del self._pred[v][u]
        except KeyError as err:
            raise nx.NetworkXError(f"The edge {u}-{v} not in graph.") from err

    def get_superclasses(self, n: Hashable, labels: Iterable[str]=[], return_type: Union[Type[List[Hashable]], Type[Set[Hashable]]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find all the direct superclasses of n, returning as either a list or a set.
        
        If labels (an iterable of acceptable labels) is specified, then the output will be limited to those that connect to n via an edge labelled by one of the given labels.
        
            This would effectively be the superclasses of n in a particular subgraph of the taxonomy specified by the edge label restrictions.
        '''     
        try:
            succ = self._succ[n]
        except KeyError as err:
            raise nx.NetworkXError(f"The node {n} is not in the digraph.") from err
        if labels != []:
            try:
                succ = {n:attr for n,attr in succ.items() if attr['label'] in labels}
            except KeyError as err:
                raise nx.NetworkXError("Node label not specified") from err
        return return_type(succ.keys())
    
    def get_subclasses(self, n: Hashable, labels:Iterable[str]=[], return_type: Union[Type[List[Hashable]], Type[Set[Hashable]]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find all the direct subclasses of n, returning as either a list or a set.
        
        For interpretation and usage of the labels parameter, see the docstring for get_superclasses.
        '''        
        try:
            pred = self._pred[n]
        except KeyError as err:
            raise nx.NetworkXError(f"The node {n} is not in the digraph.") from err
        if labels != []:
            try:
                pred = {n:attr for n,attr in pred.items() if attr['label'] in labels}
            except KeyError as err:
                raise nx.NetworkXError("Node label not specified") from err
        return return_type(pred.keys())
    
    def get_ancestors(self, node: Hashable, labels: Iterable[str]=[], return_type: Union[Type[List[Hashable]], Type[Set[Hashable]]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find all the ancestors of n, returning as either a list or a set.
        
        An ancestor is a node reachable from n by a sequence of superclass relations.
        
        The results does NOT include n itself.
        
        The labels parameter is similar to its counterpart in get_superclasses(). Here it is being used to restrict ancestorship iteratively.
        
            This would effectively be the ancestors of n in a particular subgraph of the taxonomy specified by the edge label restrictions.
        '''        
        queue = deque([node])
        visited = {node}
        answer = {}
        while queue:
            n = queue.popleft()
            for succ in self.get_superclasses(n,labels=labels):
                if succ not in visited:
                    visited.add(succ)
                    answer[succ] = True
                    queue.append(succ)
        return return_type(answer.keys())
    
    def get_descendants(self, node: Hashable, labels: Iterable[str]=[], return_type: Union[Type[List[Hashable]], Type[Set[Hashable]]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find all the descendants of n, returning as either a list or a set.
        
        A descendant is a node reachable from n by a sequence of subclass relations.
        
        The results does NOT include n itself.
        
        For interpretation and usage of the labels parameter, see the docstring for get_ancestors.
        '''                
        queue = deque([node])
        visited = {node}
        answer = {}
        while queue:
            n = queue.popleft()
            for pred in self.get_subclasses(n,labels=labels):
                if pred not in visited:
                    visited.add(pred)
                    answer[pred] = True
                    queue.append(pred)
        return return_type(answer.keys())
    
    def get_ancestors_by_depth(self, node: Hashable, max_depth:int=1, labels: Iterable[str]=[], return_type: Union[Type[List[Hashable]], Type[Set[Hashable]]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find all the ancestors of n up to max_depth steps away from n.
        
        Similar to get_ancestors.
        '''          
        queue = deque([(node,0)])
        visited = {node}
        answer = {}
        while queue:
            n,d = queue.popleft()
            if d >= max_depth:
                continue
            for succ in self.get_superclasses(n,labels=labels):
                if succ not in visited:
                    visited.add(succ)
                    answer[succ] = True
                    queue.append((succ,d+1))
        return return_type(answer.keys())
    
    # Similar to get_descendants but allow the specification of max distance to the query node.
    def get_descendants_by_depth(self, node: Hashable, max_depth: int=1, labels: Iterable[str]=[], return_type:Union[Type[List[Hashable]], Type[Set[Hashable]]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find all the descendants of n up to max_depth steps away from n.
        
        Similar to get_descendants.
        '''                  
        queue = deque([(node,0)])
        visited = {node}
        answer = {}
        while queue:
            n,d = queue.popleft()
            if d >= max_depth:
                continue
            for pred in self.get_subclasses(n,labels=labels):
                if pred not in visited:
                    visited.add(pred)
                    answer[pred] = True
                    queue.append((pred,d+1))
        return return_type(answer.keys())
    
    def subsumes(self, u: Hashable, v: Hashable, labels: Iterable[str]=[]) -> bool:
        '''
        Check if one node u subsumes another node v. Subsumption is defined as either u == v or u is an ancestor of v.
        
        For interpretation and usage of the labels parameter, see the docstring for get_ancestors.
        '''                        
        queue = deque([u])
        visited = {u}
        while queue:
            n = queue.popleft()
            if n == v:
                return True
            for pred in self.get_subclasses(n,labels=labels):
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
        return False

    def get_label(self, node: Union[Hashable, List[Hashable]]) -> Union[str, List[str]]:
        '''
        Node label getter. Inputs can be either a node or a list of nodes. In the latter case the outputs will be a list of labels.
        
        May raise KeyError.
        '''
        
        if isinstance(node,list):
            return [self._node[n]['label'] for n in node]
        return self._node[node]['label']
    
    def set_label(self, node: Union[Hashable, List[Hashable]], label: Union[str, List[str]]) -> None:
        '''
        Node label setter. Inputs can be either a node with label or a list of nodes with labels.
        
        If a queried node is missing from the taxonomy, then this node will be automatically added with its intended label and no edge attached.
        '''
        
        if isinstance(node,list):
            for i,n in enumerate(node):
                self.add_node(n, label=label[i])
        else:
            self.add_node(node, label=label)
    
    def get_edge_label(self, u: Hashable, v: Hashable) -> str:
        '''
        Edge label getter.
        
        May raise KeyError.
        '''
        
        return self._succ[u][v]['label']
    
    def set_edge_label(self, u: Hashable, v: Hashable, label: str) -> None:
        '''
        Edge label setter.
        
        If a queried edge is missing from the taxonomy, then this edge will be automatically added with its intended label.
            
            This may also lead to the addition of relevant nodes if either u or v is missing.
        '''
        self.add_edge(u,v,label=label)
    
    def reduce_subset(self, subset: Iterable[Hashable], labels: Iterable[str]=[], reverse:bool=False, return_type:Union[Type[list], Type[set]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find the minimal subset of a set of nodes by transitive subsumption reduction. Returning as either a list or a set.
        
        If reverse = False, this method will select the nodes that do not subsume any other nodes in the input (bottom nodes).
            
        Otherwise, it will select the nodes that are not subsumed by any other nodes in the input (top nodes).
        
        For interpretation and usage of the labels parameter, see the docstring for get_superclasses.
        '''
        if not subset:
            return return_type([])
        if reverse:
            func = lambda n: self.get_ancestors(n,labels=labels,return_type=set)
        else:
            func = lambda n: self.get_descendants(n,labels=labels,return_type=set)
        subset = set(subset)
        for n in subset.copy():
            if func(n).intersection(subset):
                subset.remove(n)
        return return_type(subset)
    
    def get_LCA(self, nodes: Iterable[Hashable], labels: Iterable[str]=[], return_type:Union[Type[list], Type[set]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find the Least Common Ancestors (LCA) of the input nodes (iterable with arbitrary size). Returning as either a list or a set.
        
        An LCA is any node that satisfy the following:
            
            1. It subsumes all of the input nodes (definition for Common Ancestor, CA)
            
            2. It does not subsume any other CA
        
        When the input is empty, it returns the bottom nodes of the entire taxonomy.
        
        For interpretation and usage of the labels parameter, see the docstring for get_superclasses.
        '''        
        if not nodes:
            return return_type([k for k,v in self._pred.items() if not v])
        nodes = set(nodes)
        queue = deque([(n,{n}) for n in nodes])
        colours = {n:{n} for n in nodes}
        CA = []
        N = len(nodes)
        while queue:
            n, new_colours = queue.popleft()
            colours[n] = colours[n].union(new_colours)
            if len(colours[n]) == N:
                CA.append(n)
                continue
            for succ in self.get_superclasses(n,labels=labels):
                try:
                    if colours[n].issubset(colours[succ]):
                        continue
                except KeyError:
                    colours[succ] = colours[n]
                queue.append((succ,colours[n]))
        return self.reduce_subset(CA,labels=labels,return_type=return_type)
    
    def get_GCD(self, nodes: Iterable[Hashable], labels: Iterable[str]=[], return_type:Union[Type[list], Type[set]]=list) -> Union[List[Hashable], Set[Hashable]]:
        '''
        Find the Greatest Common Descendants (GCD) of the input nodes (iterable with arbitrary size). Returning as either a list or a set.
        
        An GCD is any node that satisfy the following:
            
            1. It is subsumed by all of the input nodes (definition for Common Descendant, CD)
            
            2. It is not subsumed by any other CD
            
        When the input is empty, it returns the top nodes of the entire taxonomy.
        
        For interpretation and usage of the labels parameter, see the docstring for get_superclasses.
        '''        
        if not nodes:
            return return_type([k for k,v in self._succ.items() if not v])
        nodes = set(nodes)
        queue = deque([(n,{n}) for n in nodes])
        colours = {n:{n} for n in nodes}
        CD = []
        N = len(nodes)
        while queue:
            n, new_colours = queue.popleft()
            colours[n] = colours[n].union(new_colours)
            if len(colours[n]) == N:
                CD.append(n)
                continue
            for succ in self.get_subclasses(n,labels=labels):
                try:
                    if colours[n].issubset(colours[succ]):
                        continue
                except KeyError:
                    colours[succ] = colours[n]
                queue.append((succ,colours[n]))
        return self.reduce_subset(CD,labels=labels,reverse=True,return_type=return_type)

    def create_subgraph(self, base: Iterable[Hashable], crop_top: bool=True, force_labels: Optional[List[List[str]]]=None, strict: bool=False) -> Taxonomy:
        '''
        Create a sub-taxonomy of the current taxonomy made of nodes that are considered "above" the input nodes (base). These are by default the nodes that are not subsumed by any of the nodes in base.
        
        If crop_top = True, the subgraph will be bounded above by the LCAs of base. Otherwise the subgraph will be bounded above by the global top nodes.
        
        If force_labels (list of list of labels) is provided, the top of subgraph will always include the LCA of base w.r.t. each set of the input edge labels from force_labels.
        
            For details on the LCA w.r.t. a restricted set of labels, see the docstring for get_superclasses and get_LCA.
            
        If strict = True, the subgraph will exclude any class that do not subsume at least one base class.
        '''    
        if not base:
            return deepcopy(self)
        subgraph = Taxonomy()
        base = self.reduce_subset(base,return_type=set)
        
        # Define the upper bound of the subgraph. The lower bound has already been defined as the (minimal subset w.r.t. subsumption of) the base
        if not crop_top:
            top = self.get_GCD([])
        elif force_labels:
            top = self.get_LCA(base,return_type=set)
            for labels in force_labels:
                top = top.union(self.get_LCA(base,labels=labels,return_type=set))
            top = self.reduce_subset(top,reverse=True)
        else:
            top = self.get_LCA(base)
        
        # Compute restrictions if required
        base_descendants = {c:True for c in set.union(*[set(self.get_descendants(b)) for b in base])}
        if strict:
            base_subsumes = {c:True for c in set.union(*[set(self.get_ancestors(b)) for b in base]).union(base)}
        
        # Build the subgraph by recursively traversing downward from top
        queue = deque(top)
        while queue:
            node = queue.popleft()
            subgraph.add_node(node,label=self.get_label(node))
            if node in base:
                continue
            for sub in self.get_subclasses(node):
                if sub in base_descendants:
                    continue
                if strict:
                    if sub not in base_subsumes:
                        continue
                subgraph.add_edge(sub,node,label=self.get_edge_label(sub,node))
                queue.append(sub)
        return subgraph
    
    def transitive_reduction(self) -> Taxonomy:
        '''
        Returns the transitive reduction of a taxonomy.
        
        Based on the networkx.transitive_reduction method.
        '''
        tr = nx.transitive_reduction(self)
        tr.add_nodes_from(self.nodes(data=True))
        tr.add_edges_from((u, v, self.edges[u, v]) for u, v in tr.edges)
        return Taxonomy(tr)
    
    def link_iri(self, iri: str) -> int:
        '''
        Search for a node in the taxonomy by its IRI in an ontology. The IRI is assumed to be of the form baseiri/name#clsid.
        
        When this method extracts clsid from the iri, the clsid is always interpreted as *int* type
        '''    
        clsid = int(re.findall(reIRI,iri)[0])
        if clsid in self.nodes:
            return clsid
        else:
            raise KeyError(f'{clsid}')
    
    def to_json(self, file_path: Union[str, os.PathLike], **kwargs) -> None:
        '''
        Save the taxonomy to a JSON file at file_path.
        
        The file will the following format:
        
            Two arrays "nodes" and "edges"
            
                "nodes" contains a list of node objects. Each node object contains the following fields:
                
                    "id": The ID of the node. The root node (ID 0) will be omitted.
                    
                    "label": The name / surface form of the node.
                    
                    Any other node attributes.
                    
                "edges" contains a list of edge objects. Each edge object contains the following fields:
                
                    "src": The ID of the parent node.
                    
                    "tgt": The ID of the child node.
                    
                    Any other edge attributes.
                    
        Any keyword arguments will be passed to the json.dump() method.
        
            For instance, passing indent=4 will result in prettier formatting.  
        '''
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
            
        towrite = deepcopy(self)
        towrite.remove_node(0)
        obj = {'nodes': [{'id':n, **towrite.nodes[n]} for n in towrite.nodes()], 'edges': [{'src':e[0], 'tgt':e[1], **towrite.edges[e]} for e in towrite.edges()]}
        with open(file_path, 'w') as outf:
            outf.write(json.dumps(obj, cls=NpEncoder, **kwargs))

def from_json(file_path: Union[str, os.PathLike]) -> Taxonomy:
    '''
    Load the taxonomy from a JSON-like file_path.
    
    The file should have the following format:
    
    Two arrays "nodes" and "edges"
    
        "nodes" contains a list of node objects. Each node object contains the following fields:
        
            Mandatory field "id": The ID of the node. ID 0 is always reserved for the root node and should be avoided.
            
            Mandatory field "label": The name / surface form of the node.
            
            Any other fields will be stored as node attributes.
            
        "edges" contains a list of edge objects. Each edge object contains the following fields:
        
            Mandatory field "src": The ID of the child node.
            
            Mandatory field "tgt": The ID of the parent node.
            
            Any other fields will be stored as edge attributes.
            
    The root node will always be created and any node with no parents specified in the file will be considered children of the root. These added edges will be given the label "auto".
    '''
    with open(file=file_path) as inf:
        obj = json.load(inf)
    taxo = Taxonomy()
    for n in obj['nodes']:
        if 'id' not in n:
            raise ValueError('Missing node id')
        nid = n.pop('id')
        if 'label' not in n:
            raise ValueError(f'Missing node label for {n["id"]}')
        taxo.add_node(nid,**n)
    for e in obj['edges']:
        if 'src' not in e:
            raise ValueError('Missing edge source')
        src = e.pop('src')
        if 'tgt' not in e:
            raise ValueError('Missing edge target')
        tgt = e.pop('tgt')
        taxo.add_edge(src,tgt,**e)
    L1_nodes = taxo.get_GCD([])
    taxo.add_node(0, label='Root Concept')
    for l1 in L1_nodes:
        taxo.add_edge(l1, 0, label='auto')
    return taxo

def from_ontology(onto: o2.Ontology) -> Taxonomy:
    '''
    Extract the taxonomy from the partial order of named subsumptions (subClassOf) of an Ontology.
    
    This method works by tracking edges recursively downward from owl.Thing (which will be assigned the key 0).
    
    Extracted edges will be labelled 'original' in the taxonomy.
    
    All informations of the ontology other than the IRIs of classes, the first label of each class and the subClassOf relations, will be ignored.
    '''
    taxo = Taxonomy()
    with onto:
        visited = {}
        queue = deque([('http://www.w3.org/2002/07/owl#Thing','')])
        while queue:
            newiri,superiri = queue.popleft()
            newclass = o2.IRIS[newiri]
            superclass = o2.IRIS[superiri] if superiri else None
            visited[(newiri, superiri)] = True
            if newclass == o2.Thing:
                node_id = 0
                node_label = 'Root Concept'
            else:
                node_id = int(re.findall(reIRI,newiri)[0])
                node_label = newclass.label[0]
            taxo.add_node(node_id,label=node_label)
            if superclass:
                super_id = 0 if superclass == o2.Thing else int(re.findall(reIRI,superiri)[0])
                taxo.add_edge(node_id,super_id,label='original')
            for subclass in newclass.subclasses():
                subiri = subclass.iri
                if (subiri,newiri) not in visited:
                    queue.append((subiri,newiri))
    tr = nx.transitive_reduction(taxo)
    tr.add_nodes_from(taxo.nodes(data=True))
    taxo = Taxonomy(tr)
    return taxo