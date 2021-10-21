import numpy as np
import math
from config import cfg
from collections import defaultdict

class TSUserStruct():
    def __init__(self, uid, dim, lambda_, v_squared):
        self.uid = uid
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


class NaiveTS():
    def __init__(self, para):
        self.users = {}
        self.dimension = para['reduce_dim']
        self.v_squared = self.get_v_squared(para['R'], para['epsilon'], para['delta'])
        self.lambda_ = para['lambda']

    def get_v_squared(self, R, epsilon, delta):
        v = R * np.sqrt(24*self.dimension/epsilon * np.log(1/delta))
        return v ** 2

    def decide(self,uid,arms):
        try:
            user=self.users[uid]
        except:
            dim = arms[0].fv['t'].shape[0]
            self.users[uid]=TSUserStruct(uid,dim,self.lambda_,self.v_squared)
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

