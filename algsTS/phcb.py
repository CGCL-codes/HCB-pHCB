import numpy as np
import math
from config import cfg
from collections import defaultdict
from naive_ts import NaiveTS

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

class TSUserStruct():
    def __init__(self,uid,arms, dim,lambda_, v_squared):
        self.uid = uid
        self.arms = arms
        self.d = dim
        self.B = lambda_*np.identity(self.d)
        self.v_squared = v_squared
        self.f = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared*np.linalg.inv(self.B))
        if not cfg.random_choice:
            self.ts = NaiveTS(cfg.ts_para)

    def updateParameters(self, fv, reward):
        self.B += np.outer(fv, fv)
        self.f += fv*reward
        self.theta_hat = np.dot(np.linalg.inv(self.B), self.f)
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared*np.linalg.inv(self.B))

    def getProb(self, fv):
        return np.dot(self.theta_estimate, fv)


class pHTS():
    def __init__(self,para,root=None):
        self.root = root
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
            arms = []
            root = self.root
            for node in root.children:
                arms.append(ArmStruct(node,2))
            dim = root.emb.shape[0]
            self.users[uid]=TSUserStruct(uid,arms,dim,self.lambda_,self.v_squared)
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


