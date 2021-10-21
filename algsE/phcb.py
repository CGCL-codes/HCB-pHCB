import numpy as np
import math
from collections import defaultdict
from config import cfg,log_config
from naive_eg import NaiveEG

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

class UserStruct():
    def __init__(self,uid,arms):
        self.uid  = uid
        self.arms = arms
        if not cfg.random_choice:
            self.eg = NaiveEG()

class pHEGreedy():
    def __init__(self,epsilon,root=None):
        self.epsilon = epsilon
        self.users = {}
        self.root = root

    def EstimateReward(self,arm):
        try:
            reward = sum(arm.feedback.values())/sum(arm.vv.values())
        except:
            reward = 0
        return reward

    def decide(self,uid):
        try:
            user=self.users[uid]
        except:
            root = self.root
            arms = []
            for node in root.children:
                arms.append(ArmStruct(node,2))
            self.users[uid]=UserStruct(uid,arms)
            user=self.users[uid]
        aid = None
        max_r = float('-inf')
        if len(user.arms)<=cfg.poolsize:
            aids = list(range(len(user.arms)))
        else:
            aids = np.random.choice(len(user.arms),cfg.poolsize,replace=False)
        for index in aids:
            arm = user.arms[index]
            reward = self.EstimateReward(arm)
            if reward>max_r:
                aid = index
                max_r = reward
        if np.random.random()<self.epsilon:
            aid = np.random.randint(0,len(user.arms))
        arm_picker = user.arms[aid]
        return arm_picker,aid

    def update(self,uid,aid,arm_picker,item,feedback):
        gid = item.gid
        arm_picker.feedback[gid] += feedback
        arm_picker.vv[gid] += 1
        arm_picker.itemclick[gid] = True
        user = self.users[uid]
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






