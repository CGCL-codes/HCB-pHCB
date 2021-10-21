import math
import numpy as np
from config import cfg
from collections import defaultdict
from naive_eg import NaiveEG

class ArmStruct():
    def __init__(self,node):
        self.node = node
        self.children = []
        self.is_leaf = node.is_leaf
        self.gids = node.gids
        self.itemclick = defaultdict(bool)
        self.feedback = defaultdict(float)
        self.vv = defaultdict(int)


class UserStruct():
    def __init__(self,uid,root):
        self.uid = uid
        self.root = root
        self.path = []
        if not cfg.random_choice:
            self.eg = NaiveEG()


class HEGreedy():
    def __init__(self,epsilon,root=None):
        self.epsilon = epsilon
        self.users={}
        self.root = root
        self.total_depth = len(cfg.new_tree_file.split('_'))-1

    def EstimateReward(self,arm):
        try:
            reward = sum(arm.feedback.values())/sum(arm.vv.values())
        except:
            reward = 0
        return reward

    def build_arm_struct(self):
        arm_root = ArmStruct(self.root)
        tmp = [arm_root]
        while(len(tmp)>0):
            new_tmp = []
            for arm in tmp:
                if arm.is_leaf:
                    continue
                for child in arm.node.children:
                    arm_child = ArmStruct(child)
                    arm.children.append(arm_child)
                    new_tmp.append(arm_child)
            tmp = new_tmp
        return arm_root

    def decide(self,uid,root=None):
        try:
            user=self.users[uid]
        except:
            root = self.build_arm_struct()
            self.users[uid]=UserStruct(uid,root)
            user=self.users[uid]
        current_node = user.root
        depth = 0
        user.path = []
        while(current_node.is_leaf==False):
            children = current_node.children
            max_r = float('-inf')
            aid = None
            poolsize = int(cfg.poolsize)
            if len(children)<=poolsize:
                arms = children
            else:
                aids = np.random.choice(len(children),poolsize,replace=False)
                arms = [children[i] for i in aids]
            for index,arm in enumerate(arms):
                reward = self.EstimateReward(arm)
                if reward>max_r:
                    aid = index
                    max_r = reward
            arm_picker = arms[aid]
            user.path.append(arm_picker)
            current_node = arm_picker
            depth += 1
        return arm_picker,aid

    def update(self,uid,aid,arm_picker,item,feedback):
        gid = item.gid
        user = self.users[uid]
        path = user.path
        assert len(path)!=0
        for i,arm_picker in enumerate(path):
            arm_picker.feedback[gid] += feedback
            arm_picker.vv[gid] += 1
            arm_picker.itemclick[gid] = True
