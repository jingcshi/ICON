import os
import sys
sys.path.append(os.getcwd() + '/../..')
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import utils.taxo_utils as _taxonomy

# The following methods are used to build the subsumption data from ontology / mappings.
# Positive data are all the direct subsumptions in the ontology.

def get_positive(taxo:_taxonomy.Taxonomy):
    data = {'Subclass':[],'Superclass':[],'labels':[]}
    for c in list(taxo.nodes):
        for s in taxo.get_descendants_by_depth(c,max_depth=2):
            data['Subclass'].append(s)
            data['Superclass'].append(c)
            data['labels'].append(1)
    return data

# Neighbours refer to direct subclasses and superclasses. Uses BFS.
def get_n_hop_neighbours(taxo:_taxonomy.Taxonomy,seed,depth):
    queue = [(seed, 0)] 
    visited = set([seed])
    neighbours = []
    while queue:
        node, hop_count = queue.pop(0)
        if hop_count == depth:
            neighbours.append(node)
            continue
        if hop_count < depth:
            for neighbor in taxo.get_subclasses(node,return_type=set).union(taxo.get_superclasses(node,return_type=set)):
                if neighbor not in visited:
                    queue.append((neighbor, hop_count + 1))
                    visited.add(neighbor)
    return neighbours

# Replace superclass with a random non subsuming class.
def get_easy_negatives_from_positive(taxo:_taxonomy.Taxonomy,positive,rate):
    data = {'Subclass':[],'Superclass':[],'labels':[]}
    classes = list(taxo.nodes)
    onto_size = len(classes)
    classes_set = set(taxo.nodes)
    N = len(positive['labels'])
    with tqdm(total = N, desc='Easy negatives') as pbar:
        for i in range(N):
            sub = positive['Subclass'][i]
            ancestors = taxo.get_ancestors(sub)
            negclasses = [classes[i] for i in np.random.choice(onto_size,int(rate))]
            for neg in negclasses:
                if neg in ancestors:
                    neg = np.random.choice(list(classes_set.difference(ancestors)),1)[0]
                data['Subclass'].append(sub)
                data['Superclass'].append(neg)
                data['labels'].append(0)
            pbar.update(1)
    return data

# Replace superclass with its neighbours obtained from random walk.
def get_hard_negatives_from_positive(taxo:_taxonomy.Taxonomy,positive,rate):
    data = {'Subclass':[],'Superclass':[],'labels':[]}
    classlist = list(taxo.nodes)
    onto_size = len(classlist)
    N = len(positive['labels'])
    with tqdm(total = N, desc='Hard negatives') as pbar:
        for i in range(N):
            sub = positive['Subclass'][i]
            sup = positive['Superclass'][i]
            ancestors = taxo.get_ancestors(sub)
            negclasses = set()
            depth = 2
            candidates = set.union(*[set(get_n_hop_neighbours(taxo,sup,d+1)) for d in range(depth)]).difference(ancestors)
            while len(candidates) < rate:
                depth += 1
                candidates = candidates.union(set(get_n_hop_neighbours(taxo,sup,depth))).difference(ancestors)
            negclasses = np.random.choice(list(candidates),int(rate))
            for neg in negclasses:
                data['Subclass'].append(sub)
                data['Superclass'].append(neg)
                data['labels'].append(0)
            pbar.update(1)
    return data
            
# Subsumption data for the training of subsumption prediction model.
def build_subs_data(taxo:_taxonomy.Taxonomy,easy_neg_to_pos_rate=1,hard_neg_to_pos_rate=1,test_size=0.05):
    pos = get_positive(taxo)
    easy_neg = get_easy_negatives_from_positive(taxo,pos,easy_neg_to_pos_rate)
    hard_neg = get_hard_negatives_from_positive(taxo,pos,hard_neg_to_pos_rate)   
    dflist = [pd.DataFrame(pos),pd.DataFrame(easy_neg),pd.DataFrame(hard_neg)]
    for df in dflist:
        df['Subclass'] = df['Subclass'].apply(lambda c: taxo.get_label(c))
        df['Superclass'] = df['Superclass'].apply(lambda c: taxo.get_label(c))
    if test_size > 0:
        test_data = [df.sample(frac=test_size) for df in dflist]
        train_data = [dflist[i].drop(testdf.index) for i,testdf in enumerate(test_data)]
        return pd.concat(train_data), pd.concat(test_data)
    else:
        return pd.concat(dflist), None

if __name__ == '__main__':
    
    with open('./data_config.json') as inf:
        config = json.loads(inf.read())
        
    path = config['sub']['data_path']
    rand_seed = config['random_seed']
    eval_split_rate = config['sub']['eval_split_rate']
    easy_negative_sample_rate = config['sub']['easy_negative_sample_rate']
    hard_negative_sample_rate = config['sub']['hard_negative_sample_rate']
    config['sub']['random_seed'] = rand_seed
    
    print(f'Generating SUB model data with the following configurations:\n{config["sub"]}')
    
    data = _taxonomy.from_json(path)
    dataset_name = re.findall(r'/(\w+).json$',path)[0]
    if rand_seed != None:
        np.random.seed(rand_seed)
    
    train_data, eval_data = build_subs_data(
        data,
        easy_neg_to_pos_rate = easy_negative_sample_rate,
        hard_neg_to_pos_rate = hard_negative_sample_rate,
        test_size = eval_split_rate)
    
    now = datetime.now()
    timestr = now.strftime('%Y%m%d-%H%M')
    train_data.to_csv(f'./../../data/sub/{dataset_name}-{timestr}-train.csv',index=False)
    print(f'Training data generated and saved at /data/sub/{dataset_name}-{timestr}-train.csv')
    if eval_data is not None:
        eval_data.to_csv(f'./../../data/sub/{dataset_name}-{timestr}-eval.csv',index=False)
        print(f'Evaluation data generated and saved at /data/sub/{dataset_name}-{timestr}-eval.csv')