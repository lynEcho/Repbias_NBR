from metrics import *
import pandas as pd
import json
import argparse
import os
import sys
from fair_metrics.Run_metrics_RecSys import metric_analysis as ma
from metric_utils.groupinfo import GroupInfo
from diversity_metrics import *
import metric_utils.position as pos


#accuracy
def eval_accuracy(pred_folder, dataset, size, file, param, lamda):
    history_file = f'csvdata/{dataset}/{dataset}_train.csv'
    truth_file = f'jsondata/{dataset}_future.json'
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)
    data_history = pd.read_csv(history_file)
    a_ndcg = []
    a_recall = []
    a_hit = []
    a_repeat_ratio = []
    a_explore_ratio = []
    a_recall_repeat = []
    a_recall_explore = []
    a_hit_repeat = []
    a_hit_explore = []


    keyset_file = f'keyset/{dataset}_keyset.json'
    #rerank file
    pred_file = f'{pred_folder}/{method}_{dataset}_{size}_{param}_{lamda}.json'
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(pred_file, 'r') as f:
        data_pred = json.load(f)

    ndcg = []
    recall = []
    hit = []
    repeat_ratio = []
    explore_ratio = []
    recall_repeat = []
    recall_explore = []
    hit_repeat = []
    hit_explore = []

    for user in keyset['test']: 
        pred = data_pred[user]
        truth = data_truth[user][1]
        # print(user)
        user_history = data_history[data_history['user_id'].isin([int(user)])]
        repeat_items = list(set(user_history['item_id']))
        truth_repeat = list(set(truth)&set(repeat_items)) # might be none
        truth_explore = list(set(truth)-set(truth_repeat)) # might be none

        u_ndcg = get_NDCG(truth, pred, size)
        ndcg.append(u_ndcg)
        u_recall = get_Recall(truth, pred, size)
        recall.append(u_recall)
        u_hit = get_HT(truth, pred, size)
        hit.append(u_hit)

        u_repeat_ratio, u_explore_ratio = get_repeat_explore(repeat_items, pred, size)
        repeat_ratio.append(u_repeat_ratio)
        explore_ratio.append(u_explore_ratio)

        if len(truth_repeat)>0:
            u_recall_repeat = get_Recall(truth_repeat, pred, size)
            recall_repeat.append(u_recall_repeat)
            u_hit_repeat = get_HT(truth_repeat, pred, size)
            hit_repeat.append(u_hit_repeat)

        if len(truth_explore)>0:
            u_recall_explore = get_Recall(truth_explore, pred, size)
            u_hit_explore = get_HT(truth_explore, pred, size)
            recall_explore.append(u_recall_explore)
            hit_explore.append(u_hit_explore)
    
    a_ndcg.append(np.mean(ndcg))
    a_recall.append(np.mean(recall))
    a_hit.append(np.mean(hit))
    a_repeat_ratio.append(np.mean(repeat_ratio))
    a_explore_ratio.append(np.mean(explore_ratio))
    a_recall_repeat.append(np.mean(recall_repeat))
    a_recall_explore.append(np.mean(recall_explore))
    a_hit_repeat.append(np.mean(hit_repeat))
    a_hit_explore.append(np.mean(hit_explore))



    file.write('param: ' + str(param) +' '+ 'lamda: ' + str(lamda) +'\n')
    file.write('recall: '+ str([round(num, 4) for num in a_recall]) +' '+ str(round(np.mean(a_recall), 4)) +' '+ str(round(np.std(a_recall) / np.sqrt(len(a_recall)), 4)) +'\n')
    file.write('ndcg: '+ str([round(num, 4) for num in a_ndcg]) +' '+ str(round(np.mean(a_ndcg), 4)) +' '+ str(round(np.std(a_ndcg) / np.sqrt(len(a_ndcg)), 4)) +'\n')
    file.write('hit: '+ str([round(num, 4) for num in a_hit]) +' '+ str(round(np.mean(a_hit), 4)) +' '+ str(round(np.std(a_hit) / np.sqrt(len(a_hit)), 4)) +'\n')

    file.write('repeat ratio: '+ str([round(num, 4) for num in a_repeat_ratio]) +' '+ str(round(np.mean(a_repeat_ratio), 4)) +' '+ str(round(np.std(a_repeat_ratio) / np.sqrt(len(a_repeat_ratio)), 4)) +'\n')
    file.write('explore ratio: '+ str([round(num, 4) for num in a_explore_ratio]) +' '+ str(round(np.mean(a_explore_ratio), 4)) +' '+ str(round(np.std(a_explore_ratio) / np.sqrt(len(a_explore_ratio)), 4)) +'\n')
    file.write('repeat recall: '+ str([round(num, 4) for num in a_recall_repeat]) +' '+ str(round(np.mean(a_recall_repeat), 4)) +' '+ str(round(np.std(a_recall_repeat) / np.sqrt(len(a_recall_repeat)), 4)) +'\n')
    file.write('explore recall: '+ str([round(num, 4) for num in a_recall_explore]) +' '+ str(round(np.mean(a_recall_explore), 4)) +' '+ str(round(np.std(a_recall_explore) / np.sqrt(len(a_recall_explore)), 4)) +'\n')
    file.write('repeat hit: '+ str([round(num, 4) for num in a_hit_repeat]) +' '+ str(round(np.mean(a_hit_repeat), 4)) +' '+ str(round(np.std(a_hit_repeat) / np.sqrt(len(a_hit_repeat)), 4)) +'\n')
    file.write('explore hit: '+ str([round(num, 4) for num in a_hit_explore]) +' '+ str(round(np.mean(a_hit_explore), 4)) +' '+ str(round(np.std(a_hit_explore) / np.sqrt(len(a_hit_explore)), 4)) +'\n')


    return np.mean(a_recall)

#item fairness
def eval_fairness(pred_folder, dataset, size, file, param, lamda, pweight): 
    
    group_file = f'popularity/{dataset}_group_purchase.json'
    with open(group_file, 'r') as f:
        group_item = json.load(f)
    group_dict = dict()
    for name, item in group_item.items():
        group_dict[name] = len(item) #the number of each group
    print(group_dict)
    group = GroupInfo(pd.Series(group_dict), 'unpop', 'pop', 'popularity')

    #IAA = []         
    EEL = []            
    EED = []             
    EER = []             
    DP = []           
    EUR = []          
    RUR = []       


    keyset_file = f'keyset/{dataset}_keyset.json'
    pred_file = f'{pred_folder}/{method}_{dataset}_{size}_{param}_{lamda}.json'

    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(pred_file, 'r') as f:
        data_pred = json.load(f)


    truth_file = f'jsondata/{dataset}_future.json' # all users
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)

    rows = []
    for user_id, items in data_truth.items():
        if user_id in keyset['test']: #only evaluate test users
            for i, item_id in enumerate(items[1]):
                if item_id in group_item['pop']:
                    rows.append((user_id, item_id, 1, 'pop', 1, 0))
                else:
                    rows.append((user_id, item_id, 1, 'unpop', 0, 1))
    test_rates = pd.DataFrame(rows, columns=['user', 'item', 'rating', 'popularity', 'pop', 'unpop']) 
    
    #row = [] #relev 
    ros = [] #recs
    for user_id, items in data_pred.items():
        if user_id in keyset['test']: #only evaluate test users
            for i, item_id in enumerate(items):
                if item_id in group_item['pop']:
                    #row.append((user_id, item_id, data_rel[user_id][item_id], 'pop', i+1))
                    if item_id in data_truth[user_id][1]:
                        ros.append((user_id, item_id, i+1, 'pop', 1, 1, 0))
                    else:
                        ros.append((user_id, item_id, i+1, 'pop', 0, 1, 0))
                else:
                    #row.append((user_id, item_id, data_rel[user_id][item_id], 'unpop', i+1))
                    if item_id in data_truth[user_id][1]:
                        ros.append((user_id, item_id, i+1, 'unpop', 1, 0, 1))
                    else:
                        ros.append((user_id, item_id, i+1, 'unpop', 0, 0, 1))
    recs = pd.DataFrame(ros, columns=['user', 'item', 'rank', 'popularity', 'rating', 'pop', 'unpop']) 
    #relev = pd.DataFrame(row, columns=['user', 'item', 'score', 'popularity', 'rank']) #in line with recs

    #MA = ma(recs, test_rates, group, original_relev=relev)
    MA = ma(recs, test_rates, group)
    default_results = MA.run_default_setting(listsize=size, pweight=pweight)

    #IAA.append(default_results['IAA'])       
    EEL.append(default_results['EEL'])     
    EED.append(default_results['EED'])       
    EER.append(default_results['EER'])       
    DP.append(default_results['logDP'])          
    EUR.append(default_results['logEUR'])          
    RUR.append(default_results['logRUR'])      


    #file.write('IAA: ' + str([round(num, 4) for num in IAA]) +' '+ str(round(np.mean(IAA), 4)) +' '+ str(round(np.std(IAA) / np.sqrt(len(IAA)), 4)) +'\n')
    file.write('EEL: ' + str([round(num, 4) for num in EEL]) +' '+ str(round(np.mean(EEL), 4)) +' '+ str(round(np.std(EEL) / np.sqrt(len(EEL)), 4)) +'\n')
    file.write('EED: ' + str([round(num, 4) for num in EED]) +' '+ str(round(np.mean(EED), 4)) +' '+ str(round(np.std(EED) / np.sqrt(len(EED)), 4)) +'\n')
    #file.write('EER: ' + str([round(num, 4) for num in EER]) +' '+ str(round(np.mean(EER), 4)) +' '+ str(round(np.std(EER) / np.sqrt(len(EER)), 4)) +'\n')
    file.write('DP: ' + str([round(num, 4) for num in DP]) +' '+ str(round(np.mean(DP), 4)) +' '+ str(round(np.std(DP) / np.sqrt(len(DP)), 4)) +'\n')
    file.write('EUR: ' + str([round(num, 4) for num in EUR]) +' '+ str(round(np.mean(EUR), 4)) +' '+ str(round(np.std(EUR) / np.sqrt(len(EUR)), 4)) +'\n')
    file.write('RUR: ' + str([round(num, 4) for num in RUR]) +' '+ str(round(np.mean(RUR), 4)) +' '+ str(round(np.std(RUR) / np.sqrt(len(RUR)), 4)) +'\n')
    
    
    return EEL


def convert_to_item_cate_matrix(group_item):
    # Get unique item IDs and category IDs
    item_ids = list(set(item_id for item_ids in group_item.values() for item_id in item_ids))
    category_ids = list(group_item.keys())

    # Create an item-category matrix with zeros
    item_cate_matrix = torch.zeros((len(item_ids)+1, len(category_ids)+1), dtype=torch.float32)

    # Fill the matrix with ones where items belong to categories
    for item_id in item_ids:
        
        for category_id in category_ids:
       
            if item_id in group_item[category_id]:
                item_cate_matrix[item_id, int(category_id)] = 1.0

    return item_cate_matrix



#diversity
def eval_diversity(pred_folder, dataset, size, file, param, lamda): 
    
    group_file = f'category/{dataset}_group_category.json'
    with open(group_file, 'r') as f:
        group_item = json.load(f)
    
    ILD = []
    ETP = []
    DS = []

    keyset_file = f'keyset/{dataset}_keyset.json'
    pred_file = f'{pred_folder}/{method}_{dataset}_{size}_{param}_{lamda}.json'
    

    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(pred_file, 'r') as f:
        data_pred = json.load(f)

    test_dict = {user: data_pred[user][:size] + [0] * (size - len(data_pred[user][:size])) for user in keyset['test']}

    rank_list = torch.tensor(list(test_dict.values())) #torch.Size([user_num, size])
    

    item_cate_matrix = convert_to_item_cate_matrix(group_item)

    diversity = diversity_calculator(rank_list, item_cate_matrix)

    ILD.append(diversity['ild'])
    ETP.append(diversity['entropy'])
    DS.append(diversity['diversity_score'])

    file.write('ILD: ' + str([round(num, 4) for num in ILD]) +' '+ str(round(np.mean(ILD), 4)) +' '+ str(round(np.std(ILD) / np.sqrt(len(ILD)), 4)) +'\n')
    file.write('ETP: ' + str([round(num, 4) for num in ETP]) +' '+ str(round(np.mean(ETP), 4)) +' '+ str(round(np.std(ETP) / np.sqrt(len(ETP)), 4)) +'\n')
    file.write('DS: ' + str([round(num, 4) for num in DS]) +' '+ str(round(np.mean(DS), 4)) +' '+ str(round(np.std(DS) / np.sqrt(len(DS)), 4)) +'\n')
                
    
    return ILD
 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', type=str, required=True, help='x')
    parser.add_argument('--method', type=str, required=True, help='x')
    parser.add_argument('--dataset', type=str, required=True, help='x')
    parser.add_argument('--eval', type=str, required=True, help='x')
    parser.add_argument('--param_list', nargs='+')
    parser.add_argument('--lamda_list', nargs='+')


    args = parser.parse_args()
    pred_folder = args.pred_folder
    method = args.method
    dataset = args.dataset
    eval = args.eval 
    param_list = args.param_list
    lamda_list = args.lamda_list

    eval_file = eval + f'{method}_{dataset}_20_grid.txt'
    f = open(eval_file, 'w')
    for param in param_list:
        for lamda in lamda_list:
    
            eval_accuracy(pred_folder, dataset, 20, f, param, lamda)

            eval_fairness(pred_folder, dataset, 20, f, param, lamda, pweight='default')

            eval_diversity(pred_folder, dataset, 20, f, param, lamda)
