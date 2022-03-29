# HierarchicyBandit   

## Introduction  
This is the implementation of WSDM 2022 paper : [Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations](https://arxiv.org/abs/2110.09905)  
The reference codes for **HCB** and **pHCB**, which are based on three different base bandit algorithms. 
1. LinUCB from [A contextual-bandit approach to personalized news article recommendation](https://dl.acm.org/doi/10.1145/1772690.1772758)  
2. epsilon-Greedy [This strategy, with random exploration on an  epsilon fraction of the traffic and greedy exploitation on the rest]
3. Thompson Sampling from [Thompson Sampling for Contextual Bandits with Linear Payoffs
](http://proceedings.mlr.press/v28/agrawal13.pdf)  

## Files in the folder

- `data/`
  - `MIND/` and `TaoBao/`
     - `item_info.pkl`: processed item file, including item id, item feature and embeddings for simulator;
     - `user_info.pkl`: processed user file, including user id, and embeddings for simulator;
     - `item_info_ts.pkl`:  processed item file for Thompson sampling;
- `algs/`: implementations of PCB and pHCB based on LinUCB.
- `algsE/`:  implementations of PCB and pHCB based on epsilon-Greedy.
- `algsTS/`:  implementations of PCB and pHCB based on Thompson Sampling.   

**Note**
1. Before testing the algorithms, you should modify the settings in config.py. 
2. For thompson sampling, we provide another 16 dimensonal feature vectors to run the experiments, since it can be faster . The original feature vectors are also work with the algorithms.
3. the user_info.pkl and item_info.pkl is formated as dictionary type. 
4. The implementation of ConUCB is released at [ConUCB](https://github.com/Xiaoyinggit/ConUCB). HMAB and ICTRUCB are specical case of CB-Category and CB-Leaf.
5. Datasets can be downloads from : [MIND](https://drive.google.com/file/d/1CBgLI9qgRaKo6ZpAMq1fc3MPjpQtDdkP/view?usp=sharing) and [TaoBao](https://drive.google.com/file/d/1uWPBIHl_dkr089kCwrn0kWXFLxBR88ek/view?usp=sharing)

## Usage:  
Download the HierarchicyBandit.zip and unzip.  You will get five folders, they are `algs/`, `algsE/`, `algsTS/`, `data/`, and `logger/`.   

**Parameters:**  
The config.py file contains:
```
dataset: is the dataset used in the experiment, it can be 'MIND' or 'TaoBao';  
T: the number of rounds of each bandit algorithm;  
k: the number of items recommended to user at each round, default is 1;  
activate_num: the hyper-papamter p for pHCB;  
activate_prob: the hyper-papamter q for pHCB;  
epsilon: the epsilon value for greedy-based algorithms;  
new_tree_file: the tree file name;  
noise_scale: the standard deviation of environmental noise;  
keep_prob: sample ratio; default is 1.0, which means testing all users.
linucb_para: the hyper-parameters for linucb algorithm;
ts_para: the hyper-parameters for thompson sampling algorithm;
poolsize: the size of candidate pool;
random_choice: whether random choice an item to user;   
```   
**Environment:** python 3.6 with Anaconda
**To run the bandit codes based on LinUCB:**  
```
$ cd algs
$ python simulator_multi_process.py
```  
**To run the bandit codes based on epsilon-Greedy:**  
```
$ cd algsE
$ python simulator_multi_process.py
``` 
**To run the bandit codes based on Thompson sampling:**  
```
$ cd algsTS
$ python simulator_multi_process.py
``` 


## Citation
If you use HCB-pHCB in your research, please cite us as follows:
```
@inproceedings{10.1145/3488560.3498459,
author = {Song, Yu and Sun, Shuai and Lian, Jianxun and Huang, Hong and Li, Yu and Jin, Hai and Xie, Xing},
title = {Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations},
year = {2022},
isbn = {9781450391320},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3488560.3498459},
doi = {10.1145/3488560.3498459},
abstract = {User interest exploration is an important and challenging topic in recommender systems, which alleviates the closed-loop effects between recommendation models and user-item interactions.Contextual bandit (CB) algorithms strive to make a good trade-off between exploration and exploitation so that users' potential interests have chances to expose. However, classical CB algorithms can only be applied to a small, sampled item set (usually hundreds), which forces the typical applications in recommender systems limited to candidate post-ranking, homepage top item ranking, ad creative selection, or online model selection (A/B test). In this paper, we introduce two simple but effective hierarchical CB algorithms to make a classical CB model (such as LinUCB and Thompson Sampling) capable to explore users' interest in the entire item space without limiting to a small item set. We first construct a hierarchy item tree via a bottom-up clustering algorithm to organize items in a coarse-to-fine manner. Then we propose ahierarchical CB (HCB) algorithm to explore users' interest on the hierarchy tree. HCB takes the exploration problem as a series of decision-making processes, where the goal is to find a path from the root to a leaf node, and the feedback will be back-propagated to all the nodes in the path. We further propose aprogressive hierarchical CB (pHCB) algorithm, which progressively extends visible nodes which reach a confidence level for exploration, to avoid misleading actions on upper-level nodes in the sequential decision-making process. Extensive experiments on two public recommendation datasets demonstrate the effectiveness and flexibility of our methods.},
booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
pages = {947–956},
numpages = {10},
keywords = {recommender system, contextual bandit, interest exploration},
location = {Virtual Event, AZ, USA},
series = {WSDM '22}
}


```

