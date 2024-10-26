import numpy as np
import argparse
import json
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum
import sys

#save category of P 
def load_category_index(total_users, topk, item_cate):
    Dhelp = np.zeros((total_users, topk))
    category = []

    for key, value in item_cate.items():
        category.append(int(key))

    for uid in range(total_users):
        for j in range(topk):
            for key, value in item_cate.items():
                if P[uid][j] in value:
                    Dhelp[uid][j] = int(key)

    return Dhelp, category



def load_repeat_index(total_users, topk, user_his, users):
    Rhelp = np.zeros((total_users, topk))

    for uid in range(total_users):
        for j in range(topk):
            if P[uid][j] in user_his[users[uid]]:
                Rhelp[uid][j] = 1

    return Rhelp


#save item group info
def read_item_index(total_users, topk, no_item_groups):
    Ihelp = np.zeros((total_users, topk, no_item_groups))
    for uid in range(total_users):
        for lid in range(topk):
            # convert item_ids to item_idx
            if P[uid][lid] in pop_item_ids: #pop
                Ihelp[uid][lid][0] = 1
            elif P[uid][lid] in unpop_item_ids: #unpop
                Ihelp[uid][lid][1] = 1
    return Ihelp

def load_ranking_matrices(relevance, topk): 
    total_users = len(relevance)

    S = np.zeros((total_users, topk)) 
    P = np.zeros((total_users, topk), dtype=int) #save item id
    users = []
    
    for i, (user_id, relevance_scores) in enumerate(relevance.items()):

        users.append(user_id) 

        #all item id ranked from relevant to irrelevant
        pred_list_0 = sorted(range(len(relevance_scores)), key=lambda x: relevance_scores[x])[::-1]
        pred_list = [x for x in pred_list_0 if x != 0] #remove 0, not a item id

        S[i] = sorted(relevance_scores[1:], reverse=True)[:topk]
        P[i] = pred_list[:topk] 

    return S, P, total_users, users


#optimization

def fairness_optimisation(fairness='N', total_users=1, alpha1 = 0.0000005, lamda = 0.01, num_pop_unpop=[], size = 10):

    print(f"Runing fairness optimisation on '{fairness}', {format(alpha1, 'f')}, {format(lamda, 'f')}")
    # V1: No. of users
    # V2: No. of top items (topk)
    # V4: no. og item groups
    V1, V2, V4 = range(total_users), range(topk), range(2)

    # initiate model
    model = Model()

    W = model.addVars(V1, V2, vtype=GRB.BINARY)
    item_group = model.addVars(V4, vtype=GRB.CONTINUOUS)
    repeat = model.addVars(V1, vtype=GRB.CONTINUOUS)
   

    if fairness == 'DF':

        model.setObjective(quicksum(S[i][j] * W[i, j] for i in V1 for j in V2) - alpha1 * (item_group[0] - item_group[1]) - lamda * quicksum(repeat[i] for i in V1), GRB.MAXIMIZE)

    
   

    for i in V1:
        model.addConstr(quicksum(W[i, j] for j in V2) == size)
       

    for k in V4:
        model.addConstr(item_group[k] == quicksum(W[i, j] * Ihelp[i][j][k] for i in V1 for j in V2) / num_pop_unpop[k])

    for i in V1:

        model.addConstr(repeat[i] == quicksum(Rhelp[i][j] * W[i, j] for j in V2) / size)
       
    # optimizing
    model.optimize()
    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', W)
        fairness = model.getAttr('x', item_group)


    return solution, fairness
   

def diversity_optimisation(diversity='DS', total_users=1, epsilon1 = 0.0000005, lamda = 0.01, size = 10, category = []):

    print(f"Runing diversity optimisation on '{diversity}', {format(epsilon1, 'f')}, {format(lamda, 'f')}")
    # V1: No. of users
    # V2: No. of top items (topk)

    V1, V2 = range(total_users), range(topk)

    # initiate model
    model = Model()


    W = model.addVars(V1, V2, vtype=GRB.BINARY)
    div = model.addVars(V1, vtype=GRB.CONTINUOUS)

    C = model.addVars(V1, category, vtype=GRB.BINARY) 
    repeat = model.addVars(V1, vtype=GRB.CONTINUOUS)



    if diversity == 'DS': 
        model.setObjective((quicksum(S[i][j] * W[i, j] for i in V1 for j in V2) / size) + epsilon1 * quicksum(div[i] for i in V1) - lamda * quicksum(repeat[i] for i in V1), GRB.MAXIMIZE)

      
  
    for i in V1:
        
        model.addConstr(quicksum(W[i, j] for j in V2) == size)
  


    for i in V1:

        model.addConstr(repeat[i] == quicksum(Rhelp[i][j] * W[i, j] for j in V2) / size)



    # Add constraints to set C[i, c] = 1 if category c is assigned to i
    for i in V1:
        for c in category:
           
            model.addConstr(quicksum(W[i, j] for j in V2 if Dhelp[i][j] == c) <= size * C[i, c]) 
            model.addConstr(quicksum(W[i, j] for j in V2 if Dhelp[i][j] == c) >= C[i, c])
        



    # Calculate div[i] based on the number of unique categories assigned to i in V1
    for i in V1:
       
        model.addConstr(div[i] == quicksum(C[i, c] for c in category) / size)


    # optimizing
    model.optimize()

    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', W)
        diversity = model.getAttr('x', div)


    return solution, diversity


 
def write_results(users, size, param, lamda, opt_mode, topk, solution):
    #covert W to result file

    rerank = dict()
    
    for i, user_id in enumerate(users):

        rerank_user = []
        for j in range(topk):
            if solution[i, j] > 0.5:
                rerank_user.append(P[i][j])

        assert len(rerank_user) == size
        rerank[user_id] = [int(x) for x in rerank_user]



    #save the new results in a file
    if opt_mode == 'DS':

        file_path = f'result/{method}_{dataset}_{size}_{param}_{lamda}.json'
    elif opt_mode == 'DF':

        file_path = f'result/{method}_{dataset}_{size}_{param}_{lamda}.json'
    
    
    with open(file_path, 'w') as json_file:
        json.dump(rerank, json_file)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--pred_folder',type=str, required=True)
    

    args = parser.parse_args()
    topk = args.topk
    size = args.size
    method = args.method
    pred_folder = args.pred_folder
    
    for dataset in ['dunnhumby', 'instacart', 'tafeng']:

        cate_file = f'category/{dataset}_group_category.json'
        with open(cate_file, 'r') as f:
            item_cate = json.load(f) 

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

        relevance_file = pred_folder+'/'+dataset+'_rel0.json'
        with open(relevance_file, 'r') as f:
            relevance = json.load(f) 
        pred_file = pred_folder+'/'+dataset+'_pred0.json'
        with open(pred_file, 'r') as f:
            pred = json.load(f) 


        S, P, total_users, users = load_ranking_matrices(relevance, topk)

        Dhelp, category = load_category_index(total_users, topk, item_cate)
        Rhelp = load_repeat_index(total_users, topk, user_his, users)
        Ihelp = read_item_index(total_users=total_users, topk=topk, no_item_groups=2) 


        for opt_mode in ['DF', 'DS']:
                    
            if opt_mode == 'DF': #RAIF
                for alpha1 in [0, 0.001, 0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:
                    for lamda in [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                        solution, fairness = fairness_optimisation(fairness=opt_mode, total_users=total_users, alpha1=alpha1, lamda=lamda, num_pop_unpop=num_pop_unpop, size=size)
                        write_results(users, size, alpha1, lamda, opt_mode, topk, solution)

            elif opt_mode == 'DS': #RADiv
                for epsilon1 in [0, 0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]:
                    for lamda in [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

                        solution, diversity = diversity_optimisation(diversity=opt_mode, total_users=total_users, epsilon1=epsilon1, lamda=lamda, size=size, category=category)
                        write_results(users, size, epsilon1, lamda, opt_mode, topk, solution)

        