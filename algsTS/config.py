import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import logging
warnings.filterwarnings("ignore")



class Params():
    def free_memory(self):
        for a in dir(self):
            if not a.startswith('__') and hasattr(getattr(self, a), 'free_memory'):
                getattr(self, a).free_memory()

def config():
    cfg = Params()

    cfg.dataset         = 'TaoBao' #TaoBao or MIND
    cfg.T               = 1000
    cfg.k               = 1
    cfg.activate_num    = 10
    cfg.activate_prob   = 0.1
    cfg.epsilon         = 0.05
    cfg.new_tree_file   = '10000_100_btree_ts.pkl'
    cfg.noise_scale     = 0.001
    cfg.keep_prob       = 0.0001
    cfg.ts_para         = {'R':0.0001,'delta':0.1,'epsilon':0.05,'lambda':0.1,'dim':{'MIND':64,'TaoBao':32},'reduce_dim':16}
    cfg.poolsize        = 50
    cfg.random_choice   = False

    return cfg

def log_config(cfg,logger):
    for key, value in cfg.__dict__.items():
        logger.info('{} : {}'.format(key,value))

cfg = config()
