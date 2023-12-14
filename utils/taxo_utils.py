import re
from typing import List, Union
from collections import deque
import owlready2 as o2
import networkx as nx
import json
import numpy as np

reID = re.compile(r'\d+')
reIRI = re.compile(r'#(.*)$')

# Taxonomies are directed acyclic graphs, with additional methods implemented to facilitate taxonomic operations
# One special attribute 'label' should be used exclusively for the surface forms of taxons, as downstream algorithms depend on it
# All node keys should be int or str
# All edges (u,v) are assumed to be subClassOf relations, meaning u is a subclass / hyponym of v

class taxonomy(nx.DiGraph):
    
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
    
    # Overrides the default NetworkX add_node method, adding return values
    # Return values:
        # 0 if new node successfully inserted
        # 2 if node already exists, but the attributes are updated
        # 1 if no action could be done
    def add_node(self, node_for_adding, **attr):
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
    
    # Overrides the default NetworkX add_edge method, checking for cycles and adding return values
    # Return values:
        # 0 if new edge successfully inserted
        # 2 if edge already exists, but the attributes are updated
        # 1 if new edge not inserted because it's a self circle or would break acyclicity
    def add_edge(self, u_of_edge, v_of_edge, **attr):
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
        if u == v or v in self.get_descendants(u):
            raise TypeError('Edge not added because it would cause a cycle')
        return_value = 2 if v in self._succ[u] else 0 
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._succ[u][v] = datadict
        self._pred[v][u] = datadict
        return return_value
    
    # Overrides the default NetworkX remove_node method, adding return values
    # Return values:
        # 0 if node successfully deleted
        # 1 if node not found
    def remove_node(self, n):
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
        return 0
            
    # Overrides the default NetworkX remove_edge method, adding return values
    # Return values:
        # 0 if edge successfully deleted
        # 1 if edge not found
    def remove_edge(self, u, v):
        try:
            del self._succ[u][v]
            del self._pred[v][u]
        except KeyError as err:
            raise nx.NetworkXError(f"The edge {u}-{v} not in graph.") from err
        return 0

    # All the direct superclasses of n, returning in either list or set
        # If labels (iterable) is specified, then only return the nodes that subsume the input node via one of the given labels
    def get_superclasses(self, n, labels=None, return_type:Union[list, set]=list):
        try:
            succ = self._succ[n]
        except KeyError as err:
            raise nx.NetworkXError(f"The node {n} is not in the digraph.") from err
        if labels:
            try:
                succ = {n:attr for n,attr in succ.items() if attr['label'] in labels}
            except KeyError as err:
                raise nx.NetworkXError("Node label not specified") from err
        return return_type(succ.keys())
    
    # All the direct subclasses of n, returning in either list or set
        # If labels (iterable) is specified, then only return the nodes that are subsumed by the input node via one of the given labels
    def get_subclasses(self, n, labels=None, return_type:Union[list, set]=list):
        try:
            pred = self._pred[n]
        except KeyError as err:
            raise nx.NetworkXError(f"The node {n} is not in the digraph.") from err
        if labels:
            try:
                pred = {n:attr for n,attr in pred.items() if attr['label'] in labels}
            except KeyError as err:
                raise nx.NetworkXError("Node label not specified") from err
        return return_type(pred.keys())
    
    # Ancestors are transitive superclasses. Does NOT include the node itself
    def get_ancestors(self, node, labels=None, return_type:Union[list, set]=list):
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
    
    # Descendants are transitive subclasses. Does NOT include the node itself
    def get_descendants(self, node, labels=None, return_type:Union[list, set]=list):
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
    
    # Similar to get_ancestors but allow the specification of max distance to the query node.
    def get_ancestors_by_depth(self, node, max_depth=1, labels=None, return_type:Union[list, set]=list):
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
    def get_descendants_by_depth(self, node, max_depth=1, labels=None, return_type:Union[list, set]=list):
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
    
    # Node label getter. May raise KeyError
    def get_label(self, node:Union[int, str, List[Union[int, str]]]):
        if isinstance(node,list):
            return [self._node[n]['label'] for n in node]
        return self._node[node]['label']
    
    # Node label setter
    def set_label(self, node:Union[int, str, List[Union[int, str]]], label:Union[str, List[str]]):
        if isinstance(node,list):
            for i,n in enumerate(node):
                self.add_node(n, label=label[i])
        else:
            self.add_node(node, label=label)
    
    # Edge label getter. May raise KeyError
    def get_edge_label(self, u:Union[int, str], v:Union[int, str]):
        return self._succ[u][v]['label']
    
    # Edge label setter
    def set_edge_label(self, u:Union[int, str], v:Union[int, str], label:str):
        self.add_edge(u,v,label=label)
    
    # Find the minimal subset of a set of nodes by transitive subsumption reduction
        # If reverse = False, select the nodes that do not subsume any other nodes in the input (bottom nodes)
        # If reverse = True, select the nodes that are not subsumed by any other nodes in the input (top nodes)
    def reduce_subset(self, subset, labels=None, reverse:bool=False, return_type:Union[list, set]=list):
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
    
    # Least Common Ancestor (LCA) of the input nodes (iterable with arbitrary size), these are the nodes that:
        # Subsume all inputs (definition for Common Ancestor (CA))
        # Do not subsume any other CA
    # If input is empty, returns all the bottom nodes in the graph
    # Returns either a list or a set
    def get_LCA(self, nodes, labels=None, return_type:Union[list, set]=list):
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
    
    # Greatest Common Descendant (GCD) of the input nodes (iterable with arbitrary size), these are the nodes that:
        # Is subsumed by all inputs (definition for Common Descendant (CD))
        # Is not subsumed by any other CD
    # If input is empty, returns all the top nodes in the graph
    # Returns either a list or a set
    def get_GCD(self, nodes, labels=None, return_type:Union[list, set]=list):
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

    # Create a subgraph of the current graph that are considered 'relevant' to the input nodes (base)
        # If crop_top = True, the subgraph will be limited to the descendants of the LCA of base. Otherwise the subgraph will start from the global top nodes
        # If force_labels (list of list of labels) is provided, the top of subgraph will always include the LCA of base w.r.t. each set of the input edge labels from force_labels
        # If strict = True, the subgraph will exclude any class that do not subsume at least one base class
    def create_subgraph(self, base, crop_top:bool=True, force_labels=None, strict:bool=False):
        if not base:
            return self.copy()
        subgraph = taxonomy()
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
    
    # The iri is assumed to be of the form baseiri/name#clsid
    # This method extracts clsid from the iri, the result is clsid, or key to access the node, rather than the node itself
    def link_iri(self,iri):
        clsid = int(re.findall(reIRI,iri)[0])
        if clsid in self.nodes:
            return clsid
        else:
            raise KeyError(f'{clsid}')
        
    # Extract the taxonomy from the partial order of named subsumptions (subClassOf) of an Ontology
    # Tracks *edges* recursively downward from owl.Thing (which will be assigned the key zero)
    # Extracted edges will be labelled 'original' in the taxonomy
    # Ignore all information other than the first label of each class and subClassOf relations
    def from_ontology(onto:o2.Ontology):
        taxo = taxonomy()
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
        taxo = taxonomy(tr)
        return taxo
    
    # Load the taxonomy from a JSON-like file_path.
    # The file should have the following format:
    # Two arrays "nodes" and "edges"
        # "nodes" contains a list of node objects. Each node object contains the following fields:
            # Mandatory field "id": The ID of the node. ID 0 is always reserved for the root concept and should be avoided.
            # Mandatory field "label": The name / surface form of the node.
            # Any other fields will be stored as node attributes.
        # "edges" contains a list of edge objects. Each edge object contains the following fields:
            # Mandatory field "src": The ID of the child concept.
            # Mandatory field "tgt": The ID of the parent concept.
            # Any other fields will be stored as edge attributes.
    # The root concept will always be created and any node with no parents specified in the file will be considered children of the root. These added edges will be given the label "auto".
    def from_json(file_path):
        with open(file=file_path) as inf:
            obj = json.load(inf)
        taxo = taxonomy()
        taxo.add_node(0,label='Root Concept')
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
        L1_nodes.remove(0)
        for l1 in L1_nodes:
            taxo.add_edge(l1, 0, label='auto')
        return taxo
    
    # Save the taxonomy to a JSON file at file_path.
    # The file will the following format:
        # Two arrays "nodes" and "edges"
            # "nodes" contains a list of node objects. Each node object contains the following fields:
                # "id": The ID of the node. The root concept (ID 0) will be omitted.
                # "label": The name / surface form of the node.
                # Any other node attributes.
            # "edges" contains a list of edge objects. Each edge object contains the following fields:
                # "src": The ID of the parent concept.
                # "tgt": The ID of the child concept.
                # Any other edge attributes.
    # Any keyword arguments will be passed to the json.dump() method.
        # For instance, setting indent=4 will result in prettier formatting.   
    def to_json(self,file_path,**kwargs):
        
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
            
        self.remove_node(0)
        obj = {'nodes': [{'id':n, **self.nodes[n]} for n in self.nodes()], 'edges': [{'src':e[0], 'tgt':e[1], **self.edges[e]} for e in self.edges()]}
        with open(file_path, 'w') as outf:
            outf.write(json.dumps(obj, cls=NpEncoder, **kwargs))