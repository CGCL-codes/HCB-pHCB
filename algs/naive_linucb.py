import numpy as np
import math
from config import cfg
from collections import defaultdict

class LinUCBUserStruct():
    def __init__(self,uid,dim,init,alpha):
        self.uid = uid
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


class NaiveLinUCB():
    def __init__(self,para,init='zero'):
        self.init=init
        self.users={}
        try:
            self.alpha=para['alpha']
        except:
            self.alpha=-1

    def decide(self,uid,arms):
        try:
            user=self.users[uid]
        except:
            dim = arms[0].fv['t'].shape[0]
            self.users[uid]=LinUCBUserStruct(uid,dim,"zero",self.alpha)
            user=self.users[uid]
        aid = None
        max_r = float('-inf')
        for index,arm in enumerate(arms):
            #each item is an arm
            reward = user.getProb(arm.fv['t'])
            if reward>max_r:
                aid = index
                max_r = reward
        arm_picker = arms[aid]
        return arm_picker


    def updateParameters(self, picked_arm, reward, uid):
        self.users[uid].updateParameters(picked_arm.fv['t'],reward)

