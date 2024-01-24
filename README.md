# ICON 

ICON (**I**mplicit **CON**cept Insertion) is a self-supervised taxonomy enrichment system designed for implicit taxonomy completion.
    
ICON works by representing new concepts with combinations of existing concepts. It uses a seed to retrieve a cluster of closely related concepts, in order to zoom in on a small facet of the taxonomy. It then enumerates subsets of the cluster and uses a generative model to create a virtual concept for each subset that is expected to represent the subset's semantic union. The generated concept will go through a series of valiadations and its placement in the taxonomy will be decided by a search based on a sequence of subsumption tests. The outcome for each validated concept will be either a new concept inserted into the taxonomy, or a merger with existing concepts. The taxonomy is being updated dynamically each step.

![system-diagram](/assets/diagrams/system.png "Flowchart of ICON")

## Table of Contents

- [Dependencies](#dependencies)
- [Usage](#usage)
    - [Preliminaries](#preliminaries)
        - [Replace simcse script](#replace-simcse-script)
    - [Sub-models](#sub-models)
        - [Fine-tuning data](#fine-tuning-data)
    - [Configurations](#configurations)
    - [Running ICON](#running-icon)
    - [Interpreting the outputs](#interpreting-the-outputs)
- [File IO Format](#file-io-format)

## Dependencies

ICON depends on the following packages:
    
- `numpy`
- `owlready2`
- `networkx`
- `tqdm`
- `nltk`
        
The pipeline for training sub-models that we provide in this README further depends on the following packages:

- `torch`
- `pandas`    
- `transformers`
- `datasets`
- `evaluate`
- `info-nce-pytorch`

Furthermore, the package `simcse` is required if you wish to use the [official demonstration notebook](/demo.ipynb).

Current dependency conflicts suggest that ICON runs best with **Python 3.8**.

## Usage

### Preliminaries
    
The simplest usage of ICON is with Jupyter notebook. A walkthrough tutorial is provided at [`demo.ipynb`](/demo.ipynb). Before initialising an ICON object, make sure you have your data and three dependent sub-models.
        
- `data`: A taxonomy (`taxo_utils.Taxonomy` object, which can be loaded from json via `taxo_utils.from_json`, for details see [File IO Format](#file-io-format) or an OWL ontology (`owlready2.Ontology` object)
            
- `ret_model` (recommended signature: `ret_model(taxo: Taxonomy, query: str, k: int, *args, **kwargs) -> List[Hashable])`: Retrieve the top-*k* concepts most closely related with the query concept in a taxonomy
            
- `gen_model` (recommended signature: `gen_model(labels: List[str], *args, **kwargs) -> str)`: Generate the union label for an arbitrary set of concept labels
            
- `sub_model` (recommended signature: `sub_model(sub: Union[str, List[str]], sup: Union[str, List[str]], *args, **kwargs) -> numpy.ndarray)`: Predict whether each `sup` subsumes the corresponding `sub` given two lists of `sub` and `sup`

The sub-models are essential plug-ins for ICON. Everything above (except `ret_model` or `gen_model` if you are using ICON in a particular setting, to be explained below) will be required for ICON to function.

#### Replace simcse script

If you wish to use the `RET_model` template from `/demo.ipynb`, please temporarily replace the `tool.py` in your SimCSE directory with `/utils/replace_simcse/tool.py`. The original SimCSE package displays some loggings and progress bars that are unnecessary for ICON's purposes, and the file replacement would suppress these outputs without affecting other functionalities.
        
### Sub-models

**eBay models**: Models fine-tuned on eBay data with the pipeline described below are available at RNO: `/user/jingcshi/ICON_models/`.

We offer a quick pipeline for fine-tuning (roughly year 2020 strength) solid and well-known pretrained language models to obtain the three required models.

1. Use the scripts under `/data_wrangling` to build the training and evaluation data for each sub-model using your taxonomy (or the Google PT taxonomy placed there by default).

    1. Open terminal and `cd` to `/data_wrangling`.

    2. Adjust the data building settings by modifying `data_config.json`. A list of available settings and explanation on the data format is provided [below](#fine-tuning-data).

    3. Execute the scripts with `python ./FILENAME.py` where `FILENAME` is replaced by the name of the script you wish to run.

2. Download the pretrained language models from HuggingFace. Here we use [BERT](https://huggingface.co/bert-base-cased) for both ret_model and sub_model, and [T5](https://huggingface.co/t5-base) for gen_model.

3. Fine-tune the pretrained language models. A demonstration for fine-tuning each model can be found in the notebooks under `/model_training`. Notice that the tuned language models aren't exactly the sub-models to be called by ICON yet. An example of wrapping the models for ICON and an entire run can be found at `/demo.ipynb`.

Please note that this is only a suggestion for the sub-models and deploying later models may be able to enhance ICON performances.

#### Fine-tuning data

The `/data_wrangling/data_config.json` file contains the variable parameters for each of the dataset generation scripts that we provided:

1. **Universal parameters:**

    - `random_seed`: If set, this seed will be passed to the NumPy pseudorandom generator to ensure reproducibility.

    - `data_path`: Location of your raw data.

    - `eval_split_rate`: The ratio (acceptable range $[0,1)$) of evaluation set in the whole dataset.

2. **RET model:** The data will follow the standard format for contrastive learning that is made of $(q,p,n_1,\ldots,n_k)$ tuples. Each tuple is called a *minibatch*. $q$ is the query concept; $p$ is the positive concept, a concept similar to the query (in our case a *sibling* of the query in the taxonomy); $n_1,\ldots ,n_k$ are the negative concepts which should be concepts that are dissimilar to the query. A sample data is provided [here](/data/ret/google-eval.csv).

    - `concept_appearance_per_file`: How many times each concept in the taxonomy appears in the data.

    - `negative_per_minibatch`: $k$ in the aforementioned minibatch format.
    
3. **GEN model:** The data will be lists of semicolon-delimited concept names accompanied by the concept name of the list's *LCA* (least common ancestor) as reference. Each row is a `([PREFIX][C1];...;[Cn], [LCA])` tuple. Usually the LCA is not trivial (i.e. not the root concept) but an option exists to intentionally *corrupt* some of the lists so that the LCA becomes trivial. A sample data is provided [here](/data/gen/google-eval.csv).

    - `max_chunk_size`: Max length $(\geq 2)$ of the concept list in each row. The generated data will contain lists from length 1 to the specified number.

    - `corrupt_ratio`: The ratio (acceptable range $[0,1]$) of corrupted data rows.

    - `corrupt_patterns`: The specific ways data will be allowed to get corrupted. This parameter should be a list of distinct *pairs* of integers $(p_i,n_i)$ where $p$ is the number of uncorrupted concepts and $n$ is the number of randomly chosen concepts used for corruption. For each pair $p+n$ should be no greater than `max_chunk_size`, and $p$ should not equal 1 since that would be equivalent to $p=0$.

    - `pattern_weight`: The relative frequency of each corrupt pattern. These weights do not need to add up to 1. This parameter should have the same list length as `corrupt_patterns`.

    - `prompt_prefix`: The task prefix that will be prepended to all concept lists, used to facilitate the training of some language models. 

3. **SUB model:** The data will be $(\rm{sub},\rm{sup},\rm{ref})$ tuples. $\rm{ref}$ is 1 when $\rm{sub}$ is a sub-concept of $\rm{sup}$, and 0 vice versa. Positive data will be all the child-parent and grandchild-grandparent pairs in the dataset. Negative data (rows where $\rm{ref}=0$) will be generated in two ways: *easy* and *hard*. A sample data is provided [here](/data/sub/google-eval.csv).

    - `easy_negative_sample_rate`: The amount of easy negative rows relative to the number of positive rows. These negatives are obtained by replacing $\rm{sup}$ with a random concept.

    - `hard_negative_sample_rate`: The amount of hard negative rows relative to the number of positive rows. These negatives are obtained by replacing $\rm{sup}$ with a concept reached via graph random walk from the original $\rm{sup}$.
    
### Configurations
        
Once you are ready, initialise an ICON object with your preferred configurations. If you just want to see ICON at work, use all the default configurations by e.g. `iconobj = ICON(data=your_data, ret_model=your_ret_model, gen_model=your_gen_model, sub_model=your_sub_model)` followed by `iconobj.run()` (this will trigger auto mode, see below). A complete list of configurations is provided as follows:

- `mode`: Select one of the following
        
    - `'auto'`: The system will automatically enrich the entire taxonomy without supervision.
        
    - `'semiauto'`: The system will enrich the taxonomy with the seeds specified by user input.
        
    - `'manual'`: The system will try to place the new concepts specified by user input directly into the taxonomy. Does not require `gen_model`.
    
- `logging`: How much you want to see ICON reporting its progress. Set to 0 or `False` to suppress all logging. Set to 1 if you want to see a progress bar and some brief updates. Set to `True` if you want to hear basically everything! Other possible values for this argument include integers from 2 to 5 (5 is currently equivalent to `True`), and a list of message types.
        
- `rand_seed`: If provided, this will be passed to numpy and torch as the random seed. Use this to ensure reproducibility.
    
- `transitive_reduction`: Whether to perform transitive reduction on the outcome taxonomy, which will make sure it's in its simplest form with no redundancy.
    
- Auto mode config:
    
    - `max_outer_loop`: Maximal number of outer loops allowed.
        
- Semiauto mode config:
    
    - `semiauto_seeds`: An iterable of concepts that will be used as seed for each outer loop.
        
- Manual mode config:
    
    - `input_concepts`: An iterable of new concept labels to be placed in the taxonomy.
        
    - `manual_concept_bases`: If provided, each entry will become the search bases for the corresponding input concept.
        
    - `auto_bases`: If enabled, ICON will build the search bases for each input concept. Can speed up the search massively at the cost of search breadth. If disabled, `ret_model` will not be required.
        
- Retrieval config:
    
    - `retrieve_size`: The number of concepts `ret_model` will retrieve for each query. This will be passed to `ret_model` as the argument named `k`.
        
    - `restrict_combinations`: Whether you want restrict the subsets under consideration to those including the seed concept.
        
- Generation config:
    
    - `ignore_label`: The set of output labels that indicate the `gen_model`'s rejection to generate an union label
        
    - `filter_subsets`: Whether you want the `gen_model` to skip the subsets that have trivial LCAs. That is, the LCAs of the set form a subset of itself.
        
- Concept placement config:
    
    - Search domain constraints:
        
        - `subgraph_crop`: Whether to limit the search domain to the descendants of the LCAs of the concepts which are used to generate the new concept (referred to as search bases in this documentation).

        - `subgraph_force`: If provided (type: list of list of labels), the search domain will always include the LCAs of search bases w.r.t. the sub-taxonomy defined by the edges whose labels are in each list of the input. Will not take effect if `subgraph_crop = False`.
            
        - `subgraph_strict`: Whether to further limit the search domain to the subsumers of at least one base concept.
            
    - Search:
        
        - `threshold`: The `sub_model`'s minimal predicted probability for accepting subsumption.
            
        - `tolerance`: Maximal depth to continue searching a branch that has been rejected by `sub_model` before pruning branch.
            
        - `force_known_subsumptions`: Whether to force the search to place the new concept at least as general as the LCA of the search bases, and at least as specific as the union of the search bases. Enabling this will also force the search to stop at the search bases.
            
        - `force_prune_branches`: Whether to force the search to reject all subclasses of a tested non-superclass in superclass search, and to reject all superclasses of a tested non-subclass in subclass search. Enabling this will slow down the search if the taxonomy is roughly tree-like.
        
- Taxonomy update config:
    
    - `do_update`: Whether you would like to actually update the taxonomy. If set to `True`, running ICON will return the enriched taxonomy. Otherwise, running ICON will return the records of its predictions in a dictionary.
    
    - `eqv_score_func`: When ICON is updating taxonomies, it's sometimes necessary to estimate the likelihood of $a=b$ where $a$ and $b$ are two concepts, given the likelihoods of $a \sqsubseteq b$ (b subsumes a) and $b \sqsubseteq a$. This argument is therefore a function that crunches two probabilities together to estimate the intersection probability. It's usually fine to leave it as default, which is the multiplication operation.
    
    - `do_lexical_check`: Whether you would like to run a simple lexical screening for each new concept to see if it coincides with any existing concept. If set to `True`, ICON will have to pre-compute and cache the lexical features for each concept in the taxonomy when initialising.

### Running ICON

Once you figure out your desired configurations and have initialised an ICON object, you can run ICON by simply calling `run()`. If you want to change configurations, simply do

`iconobj.update_config(**your_new_config)`
    
For instance,

`iconobj.update_config(threshold=0.9, ignore_label=iconobj.config.gen_config.ignore_label + ['Owl:Thing'])`

Would set the subsumption prediction threshold to 0.9, and add `'Owl:Thing'` to the list of ignored generated labels.

### Interpreting the outputs

The outcome of an ICON run will either be the enriched taxonomy or a record of ICON's predictions.
In the former case, you can save a taxonomy by `your_taxo_object.to_json(your_path, **your_kwargs)`. In the latter case, the record will be a Python dictionary in the form of

    {concept_name1:
        {'eqv': eqv_1,
        'sup': sup_1,
        'sub': sub_1},
    concept_name2:
        {'eqv': eqv_2,
        'sup': sup_2,
        'sub': sub_2},
    ...
    }
Where each `eqv` is either empty or a single key-value pair `label: score` with the predicted equivalent concept and its confidence score. Likewise, each `sup` and `sub` is either empty or a dictionary of such key-value pairs, but potentially including more than one concept.

## File IO format

ICON reads and writes taxonomies in a designated JSON format. In particular, the files are expected to have:
    
Two arrays `"nodes"` and `"edges"`
    
1. `"nodes"` contains a list of node objects. Each node object contains the following fields:
            
    - Mandatory field `"id"`: The ID of the node. ID `0` is always reserved for the root node and should be avoided.
                
    - Mandatory field `"label"`: The name / surface form of the node.
                
    - Any other fields will be stored as node attributes.
                
2. `"edges"` contains a list of edge objects. Each edge object contains the following fields:
            
    - Mandatory field `"src"`: The ID of the child node.
                
    - Mandatory field `"tgt"`: The ID of the parent node.
                
    - Any other fields will be stored as edge attributes.
    
While the only attribute ICON explicitly uses for each node or edge is `"label"`, you can store other attributes, for instance node term embeddings, as additional fields. These attributes will be stored in `Taxonomy` objects. An example file can be found in the data directory [here](/data/raw/google.json).
