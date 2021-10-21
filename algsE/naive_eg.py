import numpy as np
import math
from config import cfg
from collections import defaultdict

class EGUserStruct():
    def __init__(self,uid):
        self.uid = uid
        self.feedback = defaultdict(float)
        self.vv = defaultdict(int)

    def updateParameters(self,gid,reward):
        self.feedback[gid] += reward
        self.vv[gid] += 1

    def getProb(self,gid):
        if self.vv[gid]==0:
            return 0
        return self.feedback[gid]/self.vv[gid]


class NaiveEG():
    def __init__(self):
        self.users = {}

    def decide(self,uid,arms):
        try:
            user=self.users[uid]
        except:
            self.users[uid]=EGUserStruct(uid)
            user=self.users[uid]
        aid = None
        max_r = float('-inf')
        for index,arm in enumerate(arms):
            #each item is an arm
            reward = user.getProb(arm.gid)
            if reward>max_r:
                aid = index
                max_r = reward
        eps = cfg.epsilon
        if np.random.random()>=eps:
            arm_picker = arms[aid]
        else:
            aid = np.random.randint(len(arms))
            arm_picker = arms[aid]
        return arm_picker


    def updateParameters(self, picked_arm, reward, uid):
        self.users[uid].updateParameters(picked_arm.gid,reward)

