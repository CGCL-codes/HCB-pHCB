import numpy as np
import math
from config import cfg
from collections import defaultdict
from naive_eg import NaiveEG

class ArmStruct():
    def __init__(self,arm):
        self.arm = arm
        self.gids = arm.gids
        self.feedback = 0.0
        self.vv = 0

class ESUserStruct():
    def __init__(self,uid):
        self.uid = uid
        if not cfg.random_choice:
            self.eg = NaiveEG()

    def getProb(self,arm):
        if arm.vv==0:
            return 0
        return arm.feedback/arm.vv

    def updateParameters(self, arm, reward):
        arm.vv += 1
        arm.feedback += reward


class CEG():
    def __init__(self,arms):
        self.users={}
        self.arms = [ArmStruct(arm) for arm in arms]

    def decide(self,uid):
        try:
            user=self.users[uid]
        except:
            self.users[uid]=ESUserStruct(uid)
            user=self.users[uid]
        aid = None
        max_r = float('-inf')
        if len(self.arms)<=cfg.poolsize:
            arms = self.arms
        else:
            aids = np.random.choice(len(self.arms),cfg.poolsize,replace=False)
            arms = [self.arms[i] for i in aids]
        for index,arm in enumerate(arms):
            reward = user.getProb(arm)
            if reward>max_r:
                aid = index
                max_r = reward
        eps = cfg.epsilon
        if np.random.random()>=eps:
            arm_picker = arms[aid]
        else:
            aid = np.random.randint(len(arms))
            arm_picker = arms[aid]
        return arm_picker,aid


    def updateParameters(self, picked_arm, reward, uid):
        self.users[uid].updateParameters(picked_arm,reward)

    def update(self,uid,aid,arm_picker,item,feedback):
        self.updateParameters(arm_picker,feedback,uid)

