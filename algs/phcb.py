import numpy as np
import math
from config import cfg
from collections import defaultdict
from naive_linucb import NaiveLinUCB

class ArmStruct():
    def __init__(self,arm,depth):
        self.arm = arm
        self.gids = arm.gids
        self.depth = depth # only when arm is a tree node
        self.itemclick = defaultdict(bool)
        self.feedback = defaultdict(float)
        self.vv = defaultdict(int)

    def expand(self):
        if (sum(self.vv.values()))<cfg.activate_num*np.log(self.depth):
            return False
        if (sum(self.feedback.values())/sum(self.vv.values()))<cfg.activate_prob*np.log(self.depth):
            return False
        return True

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


class pHCB():
    def __init__(self,para,init='zero',root=None):
        self.init=init
        self.users={}
        try:
            self.alpha=para['alpha']
        except:
            self.alpha=-1
        self.root = root

    def decide(self,uid,root=None):
        try:
            user=self.users[uid]
        except:
            arms = []
            root = self.root
            for node in root.children:
                arms.append(ArmStruct(node,2))
            dim = root.emb.shape[0]
            self.users[uid]=LinUCBUserStruct(uid,arms,dim,"zero",self.alpha)
            user=self.users[uid]
        aid = None
        max_r = float('-inf')
        if len(user.arms)<=cfg.poolsize:
            aids = list(range(len(user.arms)))
        else:
            aids = np.random.choice(len(user.arms),cfg.poolsize,replace=False)
        for index in aids:
            arm = user.arms[index]
            depth = arm.depth
            reward = user.getProb(arm.arm.emb)*depth
            if reward>max_r:
                aid = index
                max_r = reward
        arm_picker = user.arms[aid]
        return arm_picker,aid


    def updateParameters(self, picked_arm, reward, uid):
        self.users[uid].updateParameters(picked_arm.arm.emb,reward)

    def update(self,uid,aid,arm_picker,item,feedback):
        gid = item.gid
        arm_picker.feedback[gid] += feedback
        arm_picker.vv[gid] += 1
        arm_picker.itemclick[gid] = True
        user = self.users[uid]
        self.updateParameters(arm_picker,feedback,uid)
        if arm_picker.expand() and arm_picker.arm.is_leaf==False:
            depth = arm_picker.depth+1
            user.arms.pop(aid)
            for node in arm_picker.arm.children:
                arm = ArmStruct(node,depth)
                for gid in arm.gids:
                    arm.itemclick[gid]=arm_picker.itemclick[gid]
                    arm.feedback[gid]=arm_picker.feedback[gid]
                    arm.vv[gid]=arm_picker.vv[gid]
                user.arms.append(arm)


