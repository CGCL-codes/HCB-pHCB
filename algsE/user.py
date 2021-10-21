import numpy as np
import pickle
from config import cfg

class User():
    def __init__(self,uid,x1,x2,bias,para={}):
        self.uid     = uid
        self.fv      = {} #feature vector for training/simulator
        self.fv['t'] = x1 # training
        self.fv['s'] = x2 # simulator
        self.fv['b'] = bias # simulator bias
        self.para    = para #other info

class UserManager():
    def __init__(self,path):
        self.path    = path
        self.users   = {}
        self.n_users = 0

    def load_users(self):
        user_info = pickle.load(open(self.path,'rb'))
        for key,j_s in user_info.items():
            if np.random.random()>cfg.keep_prob:
                continue
            uid = j_s['uid']
            x1  = j_s['x1']
            x2  = j_s['x2']
            bias= j_s['bias']
            self.users[uid] = User(uid,x1,x2,bias)
        self.n_users=len(self.users)

    def load_nnexp_users(self):
        self.users = {}
        user_info = pickle.load(open(self.path,'rb'))
        user_info = dict(sorted(user_info.items(),key=lambda x:x[1]['gini'],reverse=True))
        self.n_users = int(len(user_info)*cfg.keep_prob)
        for i,uid in enumerate(user_info):
            if i>=self.n_users:
                break
            j_s = user_info[uid]
            uid = j_s['uid']
            x1  = j_s['x1']
            x2  = j_s['x2']
            bias= j_s['bias']
            his = j_s['his']
            imp = j_s['imp']
            para = {'his':his,'imp':imp}
            self.users[uid] = User(uid,x1,x2,bias,para)
        self.n_users=len(self.users)

