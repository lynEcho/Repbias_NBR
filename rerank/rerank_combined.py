import numpy as np
import argparse
import json
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum
import sys

#rep-expl
def load_ranking_matrices_rep(relevance, topk, pred, users): #design for trex-rep

    total_users = len(relevance) 

    SR = np.zeros((total_users, topk))            #save repeat item score
    PR = np.zeros((total_users, topk), dtype=int) #save repeat item id

    ind = 0
    for user in users:
        pred_list = pred[user]
        rel_list = relevance[user]
        assert len(pred_list) == len(rel_list)
        k = 0
        for j in range(len(pred_list)):
            
            PR[ind][k] = pred_list[j]  #fill 0 for history less than topk
            SR[ind][k] = rel_list[j]   #fill 0 for history less than topk
            k += 1

            if k == topk:
                break

        ind += 1

    return SR, PR, total_users

def load_ranking_matrices_expl(relevance, topk, user_history):

    total_users = len(relevance)
  
    SE = np.zeros((total_users, topk))            #save explore item score
    PE = np.zeros((total_users, topk), dtype=int) #save explore item id

    users = []
    
    for i, (user_id, relevance_scores) in enumerate(relevance.items()):
        
        users.append(user_id)
        #test_user_history_set
        list_of_lists = user_history[user_id][1:-1]
        history_set = set([item for sublist in list_of_lists for item in sublist])

        #all item id ranked from relevant to irrelevant
        pred_list_0 = sorted(range(len(relevance_scores)), key=lambda x: relevance_scores[x])[::-1]
        pred_list = [x for x in pred_list_0 if x != 0] #remove 0, not a item id

        #generate explore candidate
        t = 0
        for j in range(len(pred_list)):
            if pred_list[j] not in history_set:
                PE[i][t] = pred_list[j]
                SE[i][t] = relevance_scores[pred_list[j]]
                t += 1

            if t == topk:
                break



    return SE, PE, total_users, users

def load_category_index_rep(total_users, topk, item_cate):
    RDhelp = np.zeros((total_users, topk))
    
    for uid in range(total_users):
        for j in range(topk):
            for key, value in item_cate.items():
                if PR[uid][j] in value:
                    RDhelp[uid][j] = int(key)

    return RDhelp


def load_category_index_expl(total_users, topk, item_cate):
    EDhelp = np.zeros((total_users, topk))
    for uid in range(total_users):
        for j in range(topk):
            for key, value in item_cate.items():
                if PE[uid][j] in value:
                    EDhelp[uid][j] = int(key)

    return EDhelp

def read_item_index_rep(total_users, topk, no_item_groups):
    RIhelp = np.zeros((total_users, topk, no_item_groups)) #fill 0 for history less than topk
    for uid in range(total_users):
        for lid in range(topk):
            # convert item_ids to item_idx
            if PR[uid][lid] in pop_item_ids: #pop
                RIhelp[uid][lid][0] = 1
            elif PR[uid][lid] in unpop_item_ids: #unpop
                RIhelp[uid][lid][1] = 1
    return RIhelp

def read_item_index_expl(total_users, topk, no_item_groups):
    EIhelp = np.zeros((total_users, topk, no_item_groups))
    for uid in range(total_users):
        for lid in range(topk):
            # convert item_ids to item_idx
            if PE[uid][lid] in pop_item_ids: #pop
                EIhelp[uid][lid][0] = 1
            elif PE[uid][lid] in unpop_item_ids: #unpop
                EIhelp[uid][lid][1] = 1
    return EIhelp


def repeat_ratio(total_users, size):
    rep_rel = []

    for i in range(total_users):
        for j in range(size):
            rep_rel.append(SR[i][j])

    sorted_rep_rel = sorted(rep_rel, reverse=True)

    n = len(sorted_rep_rel)

    cutoff = []

    for i in range(1, 20):

        cutoff.append(sorted_rep_rel[int(n * 0.05 * i)])
    
    cutoffs = cutoff[::-1]
    
    return cutoffs



#optimization

def fairness_optimisation(fairness='N', total_users=1, alpha2 = 0.0000005, theta = 0.01, num_pop_unpop=[], size = 10, h = []):

    print(f"Runing fairness optimisation on '{fairness}', {format(alpha2, 'f')}, {format(theta, 'f')}")
    # V1: No. of users
    # V2: No. of top items (topk)
    # V4: no. og item groups
    V1, V2, V4 = range(total_users), range(topk), range(2)

    # initiate model
    model = Model()

    WR = model.addVars(V1, V2, vtype=GRB.BINARY)
    WE = model.addVars(V1, V2, vtype=GRB.BINARY)


   
    item_group = model.addVars(V4, vtype=GRB.CONTINUOUS)
   
  
    if fairness == 'DF':
        
        model.setObjective((quicksum(SR[i][j] * WR[i, j] for i in V1 for j in V2) + quicksum(SE[i][j] * WE[i, j] for i in V1 for j in V2)) 
                            - alpha2 * (item_group[0] - item_group[1]), GRB.MAXIMIZE)

    
    for i in V1:
        model.addConstr(quicksum(WR[i, j] for j in V2) == h[i])
        model.addConstr(quicksum(WE[i, j] for j in V2) == size - h[i])
    
    for i in V1:
        for j in V2:
            model.addConstr(WR[i, j] <= 9999999 * PR[i][j])

    for k in V4:
        model.addConstr(item_group[k] == (quicksum(WR[i, j] * RIhelp[i][j][k] for i in V1 for j in V2) + quicksum(WE[i, j] * EIhelp[i][j][k] for i in V1 for j in V2)) / num_pop_unpop[k])


       
    # optimizing
    model.optimize()
    if model.status == GRB.OPTIMAL:
        solution_rep = model.getAttr('x', WR)
        solution_expl = model.getAttr('x', WE)
        fairness = model.getAttr('x', item_group)


    return solution_rep, solution_expl, fairness


def diversity_optimisation(diversity='DS', total_users=1,  epsilon2= 0.0000005, theta = 0.01, size = 10, category = [], h=[]):

    print(f"Runing diversity optimisation on '{diversity}', {format(epsilon2, 'f')}, {format(theta, 'f')}")
    # V1: No. of users
    # V2: No. of top items (topk)

    V1, V2 = range(total_users), range(topk)

    # initiate model
    model = Model()

    WR = model.addVars(V1, V2, vtype=GRB.BINARY)
    WE = model.addVars(V1, V2, vtype=GRB.BINARY)

   
    C = model.addVars(V1, category, vtype=GRB.BINARY) 
    div = model.addVars(V1, vtype=GRB.CONTINUOUS)
    

    if diversity == 'DS': 
        model.setObjective(((quicksum(SR[i][j] * WR[i, j] for i in V1 for j in V2) + quicksum(SE[i][j] * WE[i, j] for i in V1 for j in V2)) / size) 
                            + epsilon2 * quicksum(div[i] for i in V1), GRB.MAXIMIZE)

 
    for i in V1:
        model.addConstr(quicksum(WR[i, j] for j in V2) == h[i])
        model.addConstr(quicksum(WE[i, j] for j in V2) == size - h[i])
    
    for i in V1:
        for j in V2:
            model.addConstr(WR[i, j] <= 9999999 * PR[i][j]) #not choose PR[i][j] = 0

    # Add constraints to set C[i, c] = 1 if category c is assigned to i
    for i in V1:
        for c in category:
            model.addConstr(quicksum(WR[i, j] for j in V2 if RDhelp[i][j] == c) + quicksum(WE[i, j] for j in V2 if EDhelp[i][j] == c) <= size * C[i, c]) 
            model.addConstr(quicksum(WR[i, j] for j in V2 if RDhelp[i][j] == c) + quicksum(WE[i, j] for j in V2 if EDhelp[i][j] == c) >= C[i, c])

    # Calculate div[i] based on the number of unique categories assigned to i in V1
    for i in V1:
        model.addConstr(div[i] == quicksum(C[i, c] for c in category) / size)


    # optimizing
    model.optimize()

    if model.status == GRB.OPTIMAL:
        solution_rep = model.getAttr('x', WR)
        solution_expl = model.getAttr('x', WE)
        diversity = model.getAttr('x', div)


    return solution_rep, solution_expl, diversity


 
def write_results(users, size, param, theta, opt_mode, topk, solution_rep, solution_expl):
    #covert W to result file

    rerank = dict()
    
    for i, user_id in enumerate(users):

        rerank_user = []
        for j in range(topk):
            if solution_rep[i, j] > 0.5:
                rerank_user.append(PR[i][j])

        for j in range(topk):
            if solution_expl[i, j] > 0.5:
                rerank_user.append(PE[i][j])

        assert len(rerank_user) == size
        rerank[user_id] = [int(x) for x in rerank_user]


    if opt_mode == 'DS':

        file_path = f'result/{method_rep}_{method_expl}_{dataset}_{size}_{theta}_{param}.json'

    elif opt_mode == 'DF':

        file_path = f'result/{method_rep}_{method_expl}_{dataset}_{size}_{theta}_{param}.json'
    
    
    
    with open(file_path, 'w') as json_file:
        json.dump(rerank, json_file)



def compute_h(size, theta, topk):
    h = []
    for i in range(total_users):

        cnt = 0 
        for j in range(topk):

            if SR[i][j] > theta:
                cnt += 1

        h.append(cnt)

    for i in range(len(h)):
        if h[i] > size:
            h[i] = size

    return h


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--method_rep', type=str, required=True)
    parser.add_argument('--method_expl', type=str, required=True)
    parser.add_argument('--pred_folder_rep', type=str, required=True)
    parser.add_argument('--pred_folder_expl', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--theta_list', nargs='+')
    

    args = parser.parse_args()
    topk = args.topk
    size = args.size
    method_rep = args.method_rep
    method_expl = args.method_expl
    pred_folder_rep = args.pred_folder_rep
    pred_folder_expl = args.pred_folder_expl
    dataset = args.dataset
    theta_list = args.theta_list

    
    cate_file = f'category/{dataset}_group_category.json'
    with open(cate_file, 'r') as f:
        item_cate = json.load(f)

    category = []
    for key, value in item_cate.items():
        category.append(int(key))
    
    history_file = f'jsondata/{dataset}_history.json'
    with open(history_file, 'r') as f:
        user_history = json.load(f)

    user_his = dict()
    for key, value in user_history.items():
        list_of_lists = user_history[key][1:-1]
        history_set = set([item for sublist in list_of_lists for item in sublist])

        user_his[key] = list(history_set)


    group_file = f'popularity/'+dataset+'_group_purchase.json'
    with open(group_file, 'r') as f:
        group_item = json.load(f)

    pop_item_ids = group_item['pop'] #0
    unpop_item_ids = group_item['unpop'] #1

    num_pop_unpop = [len(pop_item_ids), len(unpop_item_ids)]

    print(f"No. of pop Items: {num_pop_unpop[0]} and No. of unpop Items: {num_pop_unpop[1]}")



    #read predictions

    relevance_file_rep = pred_folder_rep+'/'+dataset+'_rel0.json'
    with open(relevance_file_rep, 'r') as f:
        relevance_rep = json.load(f) 
    pred_file_rep = pred_folder_rep+'/'+dataset+'_pred0.json'
    with open(pred_file_rep, 'r') as f:
        pred_rep = json.load(f) 



    relevance_file_expl = pred_folder_expl+'/'+dataset+'_rel0.json'
    with open(relevance_file_expl, 'r') as f:
        relevance_expl = json.load(f) 
    pred_file_expl = pred_folder_expl+'/'+dataset+'_pred0.json'
    with open(pred_file_expl, 'r') as f:
        pred_expl = json.load(f) 



    
    #explore-upcf
    SE, PE, total_users, users = load_ranking_matrices_expl(relevance_expl, topk, user_history)
    EIhelp = read_item_index_expl(total_users=total_users, topk=topk, no_item_groups=2) 
    EDhelp = load_category_index_expl(total_users, topk, item_cate)



    #repeat-trex
    SR, PR, total_users = load_ranking_matrices_rep(relevance_rep, topk, pred_rep, users)
    RIhelp = read_item_index_rep(total_users=total_users, topk=topk, no_item_groups=2) 
    RDhelp = load_category_index_rep(total_users, topk, item_cate)

    '''
    #compute theta
    cutoffs = repeat_ratio(total_users, size)

    print(cutoffs)

    rounded_cutoffs = [round(num, 5) for num in cutoffs]
    print(rounded_cutoffs)

    unique_list = []
    for item in rounded_cutoffs:
        if item not in unique_list:
            unique_list.append(item)

    if unique_list[0] != 0:
        unique_list.insert(0,0)
    print(unique_list)
    '''

    for opt_mode in ['DF', 'DS']:          
        
        if opt_mode == 'DF': #RAIF
            for theta in theta_list:
                h = compute_h(size, float(theta), topk) #compute h[i]
                for alpha2 in [0, 0.001, 0.01, 0.1, 1, 10, 20, 40, 60, 80, 100, 200]:

                    solution_rep, solution_expl, fairness = fairness_optimisation(fairness=opt_mode, total_users=total_users, alpha2=alpha2, theta=float(theta), num_pop_unpop=num_pop_unpop, size=size, h=h)
                    write_results(users, size, alpha2, float(theta), opt_mode, topk, solution_rep, solution_expl)

        elif opt_mode == 'DS': #RADiv
            for theta in theta_list:
                h = compute_h(size, float(theta), topk) #compute h[i]
                for epsilon2 in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]:
                    
                    solution_rep, solution_expl, diversity = diversity_optimisation(diversity=opt_mode, total_users=total_users, epsilon2=epsilon2, theta=float(theta), size=size, category=category, h=h)
                    write_results(users, size, epsilon2, float(theta), opt_mode, topk, solution_rep, solution_expl)
        