import os
import sys 
import random
import json
import argparse
import numpy as np
from metrics import *


def get_repeat_pred(dataset, mode, alpha, beta):
    history_file = 'jsondata/' + dataset + '_history.json'
    keyset_file = 'keyset/' + dataset + '_keyset.json'

    with open(history_file, 'r') as f:
        history_data = json.load(f)
    
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    item_num = keyset['item_num'] 
    train_user = keyset['train']

    #generate repetition score
    total = np.zeros(item_num) 
    rep = np.zeros(item_num) 
    rep_cnt = np.zeros(item_num) 
    for item_ind in range(1, item_num): 

        for user in train_user: 
            cnt = 0.0 #frequency for each user
            for bask in history_data[user][1:-1]: 
                if item_ind in bask:
                    cnt += 1
            if cnt != 0:
                total[item_ind] += 1 #users who bought item i at least once
                if cnt >= 2:
                    rep[item_ind] += 1 #users who repurchase item i 
                    rep_cnt[item_ind] += math.pow(cnt-1, alpha) #each user contributes to rep level

    if mode == 'no_item':
        rep_prob = np.ones(item_num)

    if mode == 'cnt_item':
        rep_prob = np.zeros(item_num) 
        for item_ind in range(1, item_num):
            if rep[item_ind] == 0:
                rep_prob[item_ind] = 0.0
            else:
                rep_prob[item_ind] = rep_cnt[item_ind]/total[item_ind] #equ5

        avg_rep_prob = np.mean(rep_prob)
        for item_ind in range(1, item_num):
            rep_prob[item_ind] = rep_prob[item_ind] + avg_rep_prob/(total[item_ind]+1) #equ6, +1 for "total[item_ind] = 0"
    


    test_user = keyset['test']
    
    pred_dict = dict()
    pred_rel_dict = dict()

    for user in test_user:
        interests = np.zeros(item_num)
        user_data = history_data[user][1:-1][::-1]
        bask_len = len(user_data)
        for rep_ind in range(bask_len):
            for item in user_data[rep_ind]:
                interests[item] += math.pow(beta, rep_ind) #equ7
        user_pred = interests * rep_prob #equ3
        
        pred = np.argsort(user_pred)[::-1]


        pred_item = [int(item) for item in pred if user_pred[item] > 0]
        pred_score = [user_pred[ind] for ind in pred_item]

        pred_dict[user] = pred_item
        pred_rel_dict[user] = pred_score


    with open(f'results/{dataset}_pred0.json', 'w') as f:
        json.dump(pred_dict, f)

    with open(f'results/{dataset}_rel0.json', 'w') as f:
        json.dump(pred_rel_dict, f)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="instacart")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    
    args = parser.parse_args()
    dataset = args.dataset
    alpha = args.alpha
    beta = args.beta
    
    mode = 'cnt_item'

    get_repeat_pred(dataset, mode, alpha, beta)

