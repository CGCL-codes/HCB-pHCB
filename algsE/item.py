import numpy as np
import pickle

class Item():
    def __init__(self,gid,x1,x2,bias,para={}):
        self.gid     = gid
        self.fv      = {} #feature vector for training/simulator
        self.fv['t'] = x1 #training
        self.fv['s'] = x2 #simulator
        self.fv['b'] = bias #bias
        self.para    = para #other info

class ItemManager():
    def __init__(self,path):
        self.path    = path
        self.items   = {}
        self.n_items = 0

    def load_items(self):
        item_info = pickle.load(open(self.path,'rb'))
        for key,j_s in item_info.items():
            gid = j_s['gid']
            x1  = j_s['x1']
            x2  = j_s['x2']
            bias= j_s['bias']
            para = {}
            para['category2'] = j_s['category2']
            self.items[gid] = Item(gid,x1,x2,bias,para)
        self.n_items=len(self.items)