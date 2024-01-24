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

# The following methods are used to build the seq2seq data from taxonomy.
# Obtain clusters from an taxonomy. Each cluster is a list of classes sharing the same common parent.
# For the current dataset, each cluster is the set of direct children and grandchildren of a certain class.
def get_clusters_from_taxonomy(taxo:_taxonomy.Taxonomy, shuffle=True):
    clusters = []
    for cat in taxo.nodes():
        if cat == 0:
            continue
        family = taxo.get_descendants_by_depth(cat,max_depth=2)
        if len(family) <= 1:
            continue
        if shuffle:
            np.random.shuffle(family)
        clusters.append(family)            
    return clusters

# Create "negative" samples by inserting random classes into clusters. A pattern is a tuple (npos,nneg) where npos classes are taken from a proper cluster, while nneg classes are randomly inserted.
def fill_cluster_with_neg(taxo,clusters,pattern,n):
    npos,nneg = pattern
    total = npos + nneg
    filled_clusters = []
    useable_clusters = [c for c in clusters if len(c) >= npos] if npos > 0 else clusters
    N = len(useable_clusters)
    classlist = list(taxo.nodes())
    taxo_size = taxo.number_of_nodes()
    with tqdm(total = n, desc=f'Negative pattern ({npos},{nneg})') as pbar:
        while len(filled_clusters) < n:
            randcluster = useable_clusters[int(np.random.choice(N))]
            size = len(randcluster)
            indices = np.random.choice(size,npos,replace=False) if npos > 0 else []
            randcluster = [randcluster[i] for i in indices]
            need_check = 1
            while len(randcluster) < total:
                randclass = classlist[int(np.random.choice(taxo_size))]
                if randclass in randcluster:
                    continue
                elif len(randcluster) == 0 or not need_check:
                        randcluster.append(randclass)
                elif taxo.get_LCA(randcluster + [randclass]) == [0]:
                    need_check = 0
                    randcluster.append(randclass)
                else:
                    continue
            if randcluster not in filled_clusters:
                filled_clusters.append(randcluster)
                pbar.update(1)
    return list(filled_clusters)

# Split the cluster into smaller sizes in order to keep the model input at reasonable length.
def split_cluster(cluster,max_chunk_size=3):
    total = len(cluster)
    progress = 0
    miniclusters = []
    while progress < total:
        if (total - progress) <= max_chunk_size:
            next_size = total - progress
        else:
            next_size = np.random.randint(2,max_chunk_size+1)
        miniclusters.append(cluster[progress:(progress+next_size)])
        progress += next_size  
    return miniclusters
    
# Formulate the sentences that we feed into the seq2seq model.
def build_prompt(taxo, classes, shuffle=False,prefix='summarize: '):
    if shuffle:
        np.random.shuffle(classes)
    prompt = prefix
    for c in classes:
        prompt = prompt + taxo.get_label(c) + '; '
    return prompt[:-2]

# Create a DataFrame for the raw data.
def write_to_pandas(taxo,clusters,prefix='summarize: '):
    data = {'text':[],'summary':[]}
    for cluster in clusters:
        data['summary'].append(taxo.nodes[taxo.get_LCA(cluster)[0]]['label'])
        data['text'].append(build_prompt(taxo,cluster,prefix))
    return pd.DataFrame(data)
            
# Seq2seq data for the training of text generation model.
def build_seq2seq_data(taxo,chunk_size=3,neg2pos_ratio=0.4,neg_patterns=[(0,2),(0,3),(2,1)],pattern_weight=[1,1,3],test_size=0.1,prefix='summarize: '):
    clusters = get_clusters_from_taxonomy(taxo)
    miniclusters = []
    dflist = []
    for cluster in clusters:
        miniclusters += split_cluster(cluster,chunk_size)
    dflist.append(write_to_pandas(taxo,miniclusters,prefix=prefix))
    N = len(miniclusters)
    if neg2pos_ratio > 0:
        pattern_weight = np.array(pattern_weight)
        pattern_weight = pattern_weight / np.sum(pattern_weight)
        for i, pattern in enumerate(neg_patterns):
            dflist.append(write_to_pandas(taxo,fill_cluster_with_neg(taxo,clusters,pattern,int(np.ceil(N * neg2pos_ratio * pattern_weight[i]))),prefix=prefix))
    if test_size > 0:
        test_data = [df.sample(frac=test_size) for df in dflist]
        train_data = [dflist[i].drop(testdf.index) for i,testdf in enumerate(test_data)]
        return pd.concat(train_data), pd.concat(test_data)
    else:
        test_data = None
        train_data = pd.concat(dflist)
        return train_data, test_data

if __name__ == '__main__':
    
    with open('./data_config.json') as inf:
        config = json.loads(inf.read())
        
    path = config['gen']['data_path']
    rand_seed = config['random_seed']
    eval_split_rate = config['gen']['eval_split_rate']
    max_chunk_size = config['gen']['max_chunk_size']
    corrupt_ratio = config['gen']['corrupt_ratio']
    corrupt_patterns = config['gen']['corrupt_patterns']
    pattern_weight = config['gen']['pattern_weight']
    prompt_prefix = config['gen']['prompt_prefix']
    config['gen']['random_seed'] = rand_seed
    
    print(f'Generating GEN model data with the following configurations:\n{config["gen"]}')
    
    data = _taxonomy.from_json(path)
    dataset_name = re.findall(r'/(\w+).json$',path)[0]
    if rand_seed != None:
        np.random.seed(rand_seed)
    
    train_data, eval_data = build_seq2seq_data(
        data,
        chunk_size = max_chunk_size,
        test_size = eval_split_rate,
        neg2pos_ratio = corrupt_ratio / (1 - corrupt_ratio),
        neg_patterns = corrupt_patterns,
        pattern_weight = pattern_weight,
        prefix=prompt_prefix)
    
    now = datetime.now()
    timestr = now.strftime('%Y%m%d-%H%M')
    train_data.to_csv(f'./../../data/gen/{dataset_name}-{timestr}-train.csv',index=False)
    print(f'Training data generated and saved at /data/gen/{dataset_name}-{timestr}-train.csv')
    if eval_data is not None:
        eval_data.to_csv(f'./../../data/gen/{dataset_name}-{timestr}-eval.csv',index=False)
        print(f'Evaluation data generated and saved at /data/gen/{dataset_name}-{timestr}-eval.csv')