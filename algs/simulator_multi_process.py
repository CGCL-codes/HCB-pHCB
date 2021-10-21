import os
from config import cfg,log_config
import sys
sys.path.append("..")
import numpy as np
import torch
from item import ItemManager
from user import UserManager
from hcb import HCB
from phcb import pHCB
from naive_linucb import NaiveLinUCB
from clinucb import CLinUCB
import pickle
import logging
from multiprocessing import Pool
from logger import setup_logger
thread_num=20
import random

dir_path = '../data/'
logger_name = cfg.dataset+'-'+'simulator_multi_process_linucb'
time_now = setup_logger('simulator_multi_process', logger_name, level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('simulator_multi_process')


class Tree():
    def __init__(self):
        self.emb         = None
        self.size        = 0
        self.gids        = []
        self.children    = None
        self.is_leaf     = False

class simulateExp():
    def __init__(self,users,items,tree,out_folder,test_iter):
        self.users = users
        self.items = items
        self.tree  = tree
        self.out_folder = out_folder
        self.test_iter = test_iter

    def noise(self):
        return np.random.normal(scale=cfg.noise_scale)

    def get_feedback(self,uid,gid):
        user = self.users[uid]
        item = self.items[gid]
        x = np.dot(user.fv['s'],item.fv['s']) + user.fv['b'] + item.fv['b']
        return 1/(1+np.exp(-x))

    def get_candidatePool(self):
        gids = list(self.items.keys())
        select_gids = np.random.choice(gids,cfg.poolsize, replace=False)
        arms = []
        for gid in select_gids:
            arms.append(self.items[gid])
        return arms


    def get_leafNodes(self,root):
        # current = [self.tree]
        current = [root]
        leafNodes = []
        while(len(current)>0):
            temp = []
            for node in current:
                if node.is_leaf:
                    leafNodes.append(node)
                else:
                    temp += node.children
            current = temp
        return leafNodes


    def get_categoryNodes(self):
        items = self.items
        categoryNodes = {}
        for gid in items:
            item = items[gid]
            category = item.para['category2']
            if category not in categoryNodes:
                categoryNodes[category] = Tree()
            categoryNodes[category].gids.append(gid)
        res = []
        for _,node in categoryNodes.items():
            x = []
            for gid in node.gids:
                x.append(items[gid].fv['t'])
            x = np.array(x)
            emb = np.mean(x,0)
            node.emb = emb
            res.append(node)
        return res


    def simulationPerUser(self,user,iters):
        process_id=os.getpid()
        print('[simulationPerUser] uid: %d, process_id: %d'%(user.uid, process_id))
        user_reward={}
        for iter_ in range(iters):
            cur_iter_noise = self.noise()
            user_reward[iter_] = {}
            for algname,alg in algorithms.items():
                if algname in ['naive_linucb']:
                    arms = self.get_candidatePool()
                    arm_picker = alg.decide(user.uid,arms)
                    feedback = self.get_feedback(user.uid,arm_picker.gid)
                    alg.updateParameters(arm_picker,feedback,user.uid)
                    user_reward[iter_][algname] = feedback
                    continue
                else:
                    arm_picker,aid = alg.decide(user.uid)
                if cfg.random_choice:
                    replace = True if len(arm_picker.gids)<cfg.k else False
                    gids = np.random.choice(arm_picker.gids,cfg.k,replace=replace)
                    all_feedback = 0
                    for gid in gids:
                        feedback = self.get_feedback(user.uid,gid)
                        alg.update(user.uid,aid,arm_picker,gid,feedback)
                        all_feedback += feedback
                    avg_feedback = all_feedback/len(gids)
                    user_reward[iter_][algname] = avg_feedback
                else:
                    if len(arm_picker.gids)<=cfg.poolsize:
                        arms = [self.items[gid] for gid in arm_picker.gids]
                    else:
                        arms = [self.items[gid] for gid in np.random.choice(arm_picker.gids,cfg.poolsize,replace=False).tolist()]
                    item = alg.users[user.uid].linucb.decide(user.uid,arms)
                    feedback = self.get_feedback(user.uid,item.gid)
                    alg.update(user.uid,aid,arm_picker,item,feedback)
                    alg.users[user.uid].linucb.updateParameters(item,feedback,user.uid)
                    user_reward[iter_][algname] = feedback
        return user_reward

    def run_algorithms(self,algorithms):
        print('run algorithms')
        pool= Pool(processes=thread_num)
        results=[]
        for uid, user in self.users.items():
            result=pool.apply_async(self.simulationPerUser,(user,self.test_iter))
            results.append(result)
        pool.close()
        pool.join()
        all_user_reward=[]
        for result in results:
            tmp_reward=result.get()
            all_user_reward.append(tmp_reward)

        AlgReward = {}
        for algname in algorithms:
            AlgReward[algname] = []

        for iter_ in range(self.test_iter):
            iter_avg_reward = {}
            for algname in algorithms:
                iter_avg_reward[algname] = 0
            for user_reward in all_user_reward:
                for algname in user_reward[iter_]:
                    iter_avg_reward[algname] += user_reward[iter_][algname]
            for algname in algorithms:
                AlgReward[algname].append(iter_avg_reward[algname]/len(all_user_reward))
        for algname in algorithms:
            AlgReward[algname] = np.cumsum(AlgReward[algname])
        return AlgReward

if __name__ == '__main__':
    Users = UserManager(dir_path + cfg.dataset + '/user_info.pkl')
    Items = ItemManager(dir_path + cfg.dataset + '/item_info.pkl')
    root = pickle.load(open(dir_path + cfg.dataset  + '/tree/' + cfg.new_tree_file,'rb'))
    Users.load_users()
    Items.load_items()
    dim = root.emb.shape[0]
    log_config(cfg,logger)
    logger.info('number of users = {:d}'.format(Users.n_users))
    simiExp = simulateExp(Users.users,Items.items,root,dir_path,cfg.T)
    leafNodes = simiExp.get_leafNodes(root)
    categoryNodes = simiExp.get_categoryNodes()
    algorithms = {}
    algorithms['hcb']            = HCB(cfg.linucb_para,root=root)
    algorithms['phcb']           = pHCB(cfg.linucb_para,root=root)
    algorithms['naive_linucb']   = NaiveLinUCB(cfg.linucb_para)
    algorithms['linucb_leaf']    = CLinUCB(cfg.linucb_para,leafNodes)
    algorithms['linucb_category']= CLinUCB(cfg.linucb_para,categoryNodes)
    print('current algorithms = ',algorithms.keys())
    AlgReward = simiExp.run_algorithms(algorithms)
    for iter_ in range(cfg.T):
        if iter_==0 or (iter_+1)%10==0:
            for algname in algorithms:
                logger.info('iter = {:d} alg = {} cumulate_reward = {:.3f}'.format(iter_+1,algname,AlgReward[algname][iter_]))