import os
import sys
sys.path.append(os.getcwd() + '/../..')
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
from tqdm import tqdm
import utils.taxo_utils as _taxonomy

# The following methods are used to build the contrastive data from ontology / mappings.
# Obtain clusters from an ontology. Each cluster is a list of class IDs (integers) sharing the same direct common parent.
def get_clusters_from_taxo(taxo:_taxonomy.Taxonomy):
    clusters = []
    for cat in list(taxo.nodes):
        family = taxo.get_subclasses(cat)
        if len(family) <= 1:
            continue
        clusters.append(family)            
    return clusters

# Method to sample pairs of classes from a cluster. Used to build the positive part of the contrastive data.
def cluster_sample(n:int,cover_rate:int=2):
    if n < 2:
        raise ValueError('At least 2 items are needed for contrastive sampling.')
    pairs = list(combinations(list(range(n)), 2))
    ratio, remainder = cover_rate // (n-1), cover_rate%(n-1)
    fullcover = [pair for pair in pairs for _ in range(ratio)]
    subcover = [(i,j) for (i,j) in pairs if min((i-j)%n,(j-i)%n) <= (remainder // 2)]
    if remainder%2 != 0 and n%2 == 0:
        subcover = subcover + [(i,j) for (i,j) in pairs if (i-j)%n == (n // 2)]
    return fullcover + subcover

# Calculate the number of pairs cluster_sample would return for a given cluster size and cover rate.
def num_pairs(n,cover_rate):
    ratio, remainder = cover_rate//(n-1), cover_rate%(n-1)
    full_num = ratio*n*(n-1)//2
    sub_num = n*(remainder//2)
    if remainder%2 != 0 and n%2 == 0:
        sub_num += n//2
    return full_num + sub_num

# Method to sample n random classes unrelated to the query class. Used to build the negative part of the contrastive data.
def get_negative(taxo:_taxonomy.Taxonomy,query,n):
    negative = set()
    classes = list(taxo.nodes)
    m = len(classes)
    # query_tokenset = [(t,'n') for t in query.tokenset[0].split(', ')]
    while len(negative) < n:
        randidx = int(np.random.choice(m,size=1))
        randclass = classes[randidx]
        if taxo.get_ancestors(query,return_type=set).intersection(taxo.get_ancestors(randclass,return_type=set)) == {0}:
        # rand_tokenset = [(t,'n') for t in randclass.tokenset[0].split(', ')]
        # if bc.tokenset_neg_check(query_tokenset,rand_tokenset):
            negative.add(randclass)
    return list(negative)

# Find a random subset whose sum is close to n. Used to find a subset of clusters to hold out for evaluation.
def solve_subarray_sum(arr, target):
    n = len(arr)
    randmap = np.random.permutation(n)
    arr = np.array([arr[i] for i in randmap])
    selected_indices = []
    subset = []
    setsum = 0
    for i in range(n):
        selected_indices.append(i)
        subset.append(arr[i])
        setsum += arr[i]
        if setsum >= target:
            break   
    return [randmap[i] for i in selected_indices]

# Contrastive data for the training of class retrieval model.
def build_contrastive_data(taxo:_taxonomy.Taxonomy,cover_rate=2,negs_per_batch=1,test_size=0.05):
    clusters = get_clusters_from_taxo(taxo)
    train_data = {'query_label':[],'positive_label':[],'negatives_label':[]}
    test_data = {'query_label':[],'positive_label':[],'negatives_label':[]}
    rows_per_cluster = [num_pairs(len(c),cover_rate=cover_rate) for c in clusters]
    total_rows = sum(rows_per_cluster)
    test_rows = int(total_rows * test_size)
    eval_cluster_idx = solve_subarray_sum(rows_per_cluster,test_rows) if test_size > 0 else []
    with tqdm(total = total_rows) as pbar:
        for i,cluster in enumerate(clusters):
            data_to_write = test_data if i in eval_cluster_idx else train_data
            pairs = cluster_sample(len(cluster),cover_rate=cover_rate)
            for j,k in pairs:
                query, positive = cluster[j], cluster[k]
                if np.random.random() < 0.5:
                    query, positive = positive, query
                negatives = get_negative(taxo,query,negs_per_batch)
                data_to_write['query_label'].append(taxo.get_label(query))
                data_to_write['positive_label'].append(taxo.get_label(positive))
                data_to_write['negatives_label'].append([taxo.get_label(n) for n in negatives])
                pbar.update()
    if test_size > 0:
        return pd.DataFrame(train_data), pd.DataFrame(test_data)
    else:
        return pd.DataFrame(train_data), None

if __name__ == '__main__':
    
    with open('./data_config.json') as inf:
        config = json.loads(inf.read())
        
    path = config['ret']['data_path']
    rand_seed = config['random_seed']
    eval_split_rate = config['ret']['eval_split_rate']
    concept_appearance_per_file = config['ret']['concept_appearance_per_file']
    negative_per_minibatch = config['ret']['negative_per_minibatch']
    config['ret']['random_seed'] = rand_seed
    
    print(f'Generating RET model data with the following configurations:\n{config["ret"]}')
    
    data = _taxonomy.from_json(path)
    dataset_name = re.findall(r'/(\w+).json$',path)[0]
    if rand_seed != None:
        np.random.seed(rand_seed)
    
    train_data, eval_data = build_contrastive_data(
        data,
        cover_rate = concept_appearance_per_file,
        negs_per_batch = negative_per_minibatch,
        test_size = eval_split_rate)
    
    now = datetime.now()
    timestr = now.strftime('%Y%m%d-%H%M')
    train_data.to_csv(f'./../../data/ret/{dataset_name}-{timestr}-train.csv',index=False)
    print(f'Training data generated and saved at /data/ret/{dataset_name}-{timestr}-train.csv')
    if eval_data is not None:
        eval_data.to_csv(f'./../../data/ret/{dataset_name}-{timestr}-eval.csv',index=False)
        print(f'Evaluation data generated and saved at /data/ret/{dataset_name}-{timestr}-eval.csv')