import numpy as np
import math
from config import cfg
from collections import defaultdict
from naive_linucb import NaiveLinUCB


class LinUCBBaseStruct():
    def __init__(self,dim,init,alpha):
        self.dim = dim
        self.A = np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))  
        if init!='zero':
            self.theta = np.random.rand(self.dim)
        else:
            self.theta = np.zeros(self.dim)
        self.alpha = alpha # store the new alpha calcuated in each iteratioin

    def getProb(self,fv):
        if self.alpha==-1:
            raise AssertionError
        mean=np.dot(self.theta.T,fv)
        var=np.sqrt(np.dot(np.dot(fv.T,self.Ainv),fv))
        pta=mean+self.alpha*var
        return pta

    def getInv(self, old_Minv, nfv):
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv

    def updateParameters(self, a_fv, reward):
        self.A+=np.outer(a_fv, a_fv)
        self.b+=a_fv*reward
        self.Ainv=self.getInv(self.Ainv,a_fv)
        self.theta=np.dot(self.Ainv, self.b)


class LinUCBUserStruct():
    def __init__(self,uid,root,dim,init,alpha):
        self.uid = uid
        self.root = root
        self.base_linucb = {}
        self.path = []
        if not cfg.random_choice:
            self.linucb = NaiveLinUCB(cfg.linucb_para)


class HCB():
    def __init__(self,para,init='zero',root=None):
        self.init=init
        self.users={}
        try:
            self.alpha=para['alpha']
        except:
            self.alpha=-1
        self.root = root
        self.total_depth = len(cfg.new_tree_file.split('_'))-1


    def decide(self,uid,root=None):
        try:
            user=self.users[uid]
        except:
            self.dim = self.root.emb.shape[0]
            self.users[uid]=LinUCBUserStruct(uid,self.root,self.dim,"zero",self.alpha)
            user=self.users[uid]
        current_node = user.root
        depth = 0
        user.path = []
        while(current_node.is_leaf==False):
            children = current_node.children
            max_r = float('-inf')
            aid = None
            if depth not in user.base_linucb:
                user.base_linucb[depth] = LinUCBBaseStruct(self.dim,"zero",self.alpha)
            poolsize = int(cfg.poolsize/self.total_depth)
            if len(children)<=poolsize:
                arms = children
            else:
                aids = np.random.choice(len(children),poolsize,replace=False)
                arms = [children[i] for i in aids]
            for index,arm in enumerate(arms):
                reward = user.base_linucb[depth].getProb(arm.emb)
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
            user.base_linucb[depth].updateParameters(arm_picker.emb,feedback)
