import numpy as np
import math
from config import cfg
from collections import defaultdict
from naive_ts import NaiveTS


class TSBaseStruct():
    def __init__(self,dim,lambda_, v_squared):
        self.d = dim
        self.B = lambda_*np.identity(self.d)
        self.v_squared = v_squared
        self.f = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared*np.linalg.inv(self.B))

    def updateParameters(self, fv, reward):
        self.B += np.outer(fv, fv)
        self.f += fv*reward
        self.theta_hat = np.dot(np.linalg.inv(self.B), self.f)
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared*np.linalg.inv(self.B))

    def getProb(self, fv):
        return np.dot(self.theta_estimate, fv)


class TSUserStruct():
    def __init__(self,uid,root):
        self.uid = uid
        self.root = root
        self.base_ts = {}
        self.path = []
        if not cfg.random_choice:
            self.ts = NaiveTS(cfg.ts_para)


class HTS():
    def __init__(self,para,root=None):
        self.root = root
        self.total_depth = len(cfg.new_tree_file.split('_'))-1
        self.users={}
        self.dimension = para['dim'][cfg.dataset]
        self.v_squared = self.get_v_squared(para['R'], para['epsilon'], para['delta'])
        self.lambda_ = para['lambda']

    def get_v_squared(self, R, epsilon, delta):
        v = R * np.sqrt(24*self.dimension/epsilon * np.log(1/delta))
        return v ** 2


    def decide(self,uid,root=None):
        try:
            user=self.users[uid]
        except:
            self.dim = self.root.emb.shape[0]
            self.users[uid]=TSUserStruct(uid,self.root)
            user=self.users[uid]
        current_node = user.root
        depth = 0
        user.path = []
        while(current_node.is_leaf==False):
            children = current_node.children
            max_r = float('-inf')
            aid = None
            if depth not in user.base_ts:
                user.base_ts[depth] = TSBaseStruct(self.dim,self.lambda_,self.v_squared)
            poolsize = int(cfg.poolsize/self.total_depth)
            if len(children)<=poolsize:
                arms = children
            else:
                aids = np.random.choice(len(children),poolsize,replace=False)
                arms = [children[i] for i in aids]
            for index,arm in enumerate(arms):
                reward = user.base_ts[depth].getProb(arm.emb)
                if reward>max_r:
                    aid = index
                    max_r = reward
            arm_picker = arms[aid]
            user.path.append(arm_picker)
            current_node = arm_picker
            depth += 1
        return arm_picker,aid

    def update(self,uid,aid,arm_picker,item,feedback):
        user = self.users[uid]
        path = user.path
        assert len(path)!=0
        for i,arm_picker in enumerate(path):
            depth = i
            user.base_ts[depth].updateParameters(arm_picker.emb,feedback)
