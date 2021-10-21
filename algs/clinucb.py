import numpy as np
import math
from config import cfg
from collections import defaultdict
from naive_linucb import NaiveLinUCB

class ArmStruct():
    def __init__(self,arm):
        self.arm = arm
        self.gids = arm.gids

class LinUCBUserStruct():
    def __init__(self,uid,arms,dim,init,alpha):
        self.uid = uid
        self.arms = arms
        self.dim = dim
        self.A = np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))  
        if init!='zero':
            self.theta = np.random.rand(self.dim)
        else:
            self.theta = np.zeros(self.dim)
        self.alpha = alpha # store the new alpha calcuated in each iteratioin
        if not cfg.random_choice:
            self.linucb = NaiveLinUCB(cfg.linucb_para)



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


class CLinUCB():
    def __init__(self,para,arms,init='zero'):
        self.init=init
        self.users={}
        self.arms = [ArmStruct(arm) for arm in arms]
        try:
            self.alpha=para['alpha']
        except:
            self.alpha=-1

    def decide(self,uid):
        try:
            user=self.users[uid]
        except:
            dim = self.arms[0].arm.emb.shape[0]
            self.users[uid]=LinUCBUserStruct(uid,self.arms,dim,"zero",self.alpha)
            user=self.users[uid]
        aid = None
        max_r = float('-inf')
        if len(user.arms)<=cfg.poolsize:
            arms = user.arms
        else:
            aids = np.random.choice(len(user.arms),cfg.poolsize,replace=False)
            arms = [user.arms[i] for i in aids]
        for index,arm in enumerate(arms):
            reward = user.getProb(arm.arm.emb)
            if reward>max_r:
                aid = index
                max_r = reward
        arm_picker = arms[aid]
        return arm_picker,aid


    def updateParameters(self, picked_arm, reward, uid):
        self.users[uid].updateParameters(picked_arm.arm.emb,reward)

    def update(self,uid,aid,arm_picker,item,feedback):
        self.updateParameters(arm_picker,feedback,uid)

