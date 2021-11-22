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
