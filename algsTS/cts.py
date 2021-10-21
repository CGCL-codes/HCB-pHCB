import numpy as np
import math
from config import cfg
from collections import defaultdict
from naive_ts import NaiveTS

class ArmStruct():
    def __init__(self,arm):
        self.arm = arm
        self.gids = arm.gids

class TSUserStruct():
    def __init__(self,uid,dim,lambda_, v_squared):
        self.uid = uid
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


class CTS():
    def __init__(self,para,arms):
        self.users={}
        self.arms = [ArmStruct(arm) for arm in arms]
        self.dimension = self.arms[0].arm.emb.shape[0]
        self.v_squared = self.get_v_squared(para['R'], para['epsilon'], para['delta'])
        self.lambda_ = para['lambda']

    def get_v_squared(self, R, epsilon, delta):
        v = R * np.sqrt(24*self.dimension/epsilon * np.log(1/delta))
        return v ** 2

    def decide(self,uid):
        try:
            user=self.users[uid]
        except:
            dim = self.arms[0].arm.emb.shape[0]
            self.users[uid]=TSUserStruct(uid,dim,self.lambda_,self.v_squared)
            user=self.users[uid]
        aid = None
        max_r = float('-inf')
        if len(self.arms)<=cfg.poolsize:
            arms = self.arms
        else:
            aids = np.random.choice(len(self.arms),cfg.poolsize,replace=False)
            arms = [self.arms[i] for i in aids]
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

