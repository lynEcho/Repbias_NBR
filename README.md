# Source Code and Appendix for "Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation"

This paper proposes a model-agnostic repeat-bias-aware diversity optimization (RADiv) algorithm and a repeat-bias-aware item fairness optimization (RAIF) algorithm based on ILP for next basket recommendation (NBR).

We additionally:

* reproduce 5 NBR methods and apply the proposed algorithms to them.
* save the additional experiment results in appendix folder.


## Required packages

To run our rerank scripts, gurobipy, Numpy and Python >= 3.10 are required.

To run the published NBR methods' code, please go to the original repository and check the required packages.

## Dataset

* Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis/data
* Dunnhumby: https://www.dunnhumby.com/source-files/
* Tafeng: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset

We provide the preprocessed dataset with different formats (csvdata, jsondata, mergedata), which can be used directly.

### Format description of preprocessed dataset
* csvdata:
> user_id, order_number, item_id, basket_id

* jsondata: --> TIFUKNN, DNNTSP, DREAM, TREx

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }

> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: --> UP-CF

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}

### Format description of predicted results
* Predicted items:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}



## Code structure

* csvdata, jsondata, mergedata: contain different dataset formats.
* rerank: RADiv and RAIF algorithms for unified NBR methods and combined NBR methods.
* evaluate: scripts for evaluation.
    * fair_metrics: the fairness metrics.
    * diversity_metrics.py: the diversity metrics. 
    * metrics.py: the accuracy metrics.
    * evaluate_performance.py: evaluate the fairness, diversity and accuracy of recommendation results.  
* methods: contains 5 NBR methods.


## Pipeline

* Step 1. Run NBR methods and save the results. (Note that we use the original implementations of the authors, so we provide the original repository links, which contain the instructions of the environment setting, how to run each method, etc. We also provide our additional instructions and hyperparameters in the following section, which can make the running easier.)


* Step 2. Re-ranking: apply RADiv and RAIF to re-rank the recommendation obtained from NBR methods.


* Step 3. Evaluation: Use the evaluation scripts to get the performance before and after re-ranking.


## Re-ranking

For unified NBR methods, such as UP-CF@r, TIFUKNN, Dream, DNNTSP
```
python rerank_unified.py --topk 100 --size 20 --method XXX --pred_folder XXX

```
The re-ranking result is saved in file 'result/{method}_{dataset}_{size}_{param}_{lamda}.json'


For combined NBR methods, such as TREx:
```
python rerank_combined.py --topk 100 --size 20 --method_rep trex --method_expl upcf --dataset XXX --pred_folder_rep XXX --pred_folder_expl XXX --theta_list XXX

```
The re-ranking result is saved in 'result/{method_rep}_{method_expl}_{dataset}_{size}_{theta}_{param}.json'

### Hyperparameters for each NBR method on each dataset:

We list the hyperparameters selected from validation set, and you could directly use these values to reproduce the main experiment.

RADiv:
|            | Instacart  |  Instacart | Dunnhumby  |  Dunnhumby | TaFeng     |   TaFeng   |
|------------|------------|------------|------------|------------|------------|------------|
|            | epsilon_1  | lambda     |epsilon_1   | lambda     |epsilon_1   | lambda     |
| UP-CF@r    |   0.2      |   0.01     |  0.08      |   0.01     |   0.08     |   0.001    |
| TIFUKNN    |   0.2      |  0.01      |  0.04      |  0.001     |   0.01     |  0.001     |
| Dream      |   0.18     |   0.3      |  0.12      |   0.01     |   0.1      |    0.1     |
| DNNTSP     |    0.2     |    0.01    |   0.12     |    0.01    |   0.01     |    0.01    |
|            | epsilon_2  | theta      |epsilon_2   | theta      |epsilon_2   | theta      |
| TREx       |    0.5     |    0.34249 |   10       |    0.57303 |   0.001    |    0.12545 |



RAIF:
|            | Instacart  | Instacart  | Dunnhumby  |  Dunnhumby | TaFeng     |    TaFeng  |
|------------|------------|------------|------------|------------|------------|------------|
|            | alpha_1    | lambda     |alpha_1     | lambda     |alpha_1     | lambda     |
| UP-CF@r    |   200      |   0.9      |  50        |   0.3      |   100      |   0.8      |
| TIFUKNN    |   200      |  0.8       |  1         |  0.001     |   20       |  0.1       |
| Dream      |   200      |   0.3      |  200       |   0.5      |   20       |    0.2     |
| DNNTSP     |   200      |    0.9     |   200      |    0.4     |   200      |    0.6     |
|            | alpha_2    | theta      |alpha_2     | theta      |alpha_2     | theta      |
| TREx       |    200     |    0.4039  |  200       |    0.62322 |   200      |    0.0501  |


## Evaluation 

```

python evaluate_performance.py --pred_folder XXX --eval XXX --param_list XXX --lamda_list XXX --method XXX --dataset XXX

```

--pred_folder is the folder where you put the results after re-ranking, --eval is the folder where you save the evaluation results.



## Guidelines for NBR methods


Our reproducibility relies as much as possible on the artifacts provided by the user themselves, the following repositories have information about how to run each NBR method and the required packages.
* UP-CF@r: https://github.com/MayloIFERR/RACF
* TIFUKNN: https://github.com/HaojiHu/TIFUKNN
* DREAM: https://github.com/yihong-chen/DREAM
* DNNTSP: https://github.com/yule-BUAA/DNNTSP
* TREx: https://github.com/lynEcho/TREX

We also provide our additional instructions if the original repository is not clear, as well as the hyperparameters we use.

We set random seed as 12345, the corresponding number is 0.

Please create a folder "results" under each method to store the predicted files.


### UP-CF@r
UP-CF@r is under the folder "methods/upcf".
* Step 1: Check the dataset path and keyset path.
* Step 2: Predict and save the results using the following commands:
```
python racf.py --dataset instacart --recency 5 --asymmetry 0.25 --locality 5 --seed 12345 --number 0
...
python racf.py --dataset dunnhumby --recency 25 --asymmetry 0.25 --locality 5 --seed 12345 --number 0
...

``` 
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json

### TIFUKNN
TIFUKNN is under the folder "methods/tifuknn"
* Step 1: Predict and save the results using the following commands:
```
python tifuknn_new.py ../jsondata/instacart_history.json ../jsondata/instacart_future.json ../keyset/instacart_keyset.json 900 0.9 0.6 0.7 3 20 
...
python tifuknn_new.py ../jsondata/dunnhumby_history.json ../jsondata/dunnhumby_future.json ../keyset/dunnhumby_keyset.json 100 0.9 0.9 0.1 7 20 
...

```
Predicted file name: {dataset}_pred0.json, {dataset}_rel0.json

### Dream
Dream is under the folder "methods/dream".
* Step 1: Check the file path of the dataset in the config-param file "{dataset}conf.json"
* Step 2: Train and save the model using the following commands:
```
python trainer.py --dataset instacart --attention 1 --seed 12345 
...
python trainer.py --dataset dunnhumby --attention 1 --seed 12345 
...

```
* Step 3: Predict and save the results using the following commands:
```
python pred_results.py --dataset instacart --attention 1 --seed 12345 --number 0
...
python pred_results.py --dataset dunnhumby --attention 1 --seed 12345 --number 0
...

```
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json


### DNNTSP
DNNTSP is under the folder "methods/dnntsp".
* Step 1: Confirm the name of config-param file "{dataset}config.json" in ../utils/load_config.py. Check the file path of the dataset in the corresponding file "../utils/{dataset}conf.json". For example:
```
abs_path = os.path.join(os.path.dirname(__file__), "instacartconfig.json")
with open(abs_path) as file:
    config = json.load(file)
```
```
{
    "data": "Instacart",
    "save_model_folder": "DNNTSP",
    "history_path": "../jsondata/instacart_history.json",
    "future_path": "../jsondata/instacart_future.json",
    "keyset_path": "../keyset/instacart_keyset_0.json",
    "items_total": 29399,
    "item_embed_dim": 16,
    "cuda": 0,
    "loss_function": "multi_label_soft_loss",
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optim": "Adam",
    "weight_decay": 0
}
```
* Step 2: Train and save the models using the following command:
```
python train_main.py --seed 12345
```
* Step 3: Predict and save results using the following commands:
```
python pred_results.py --dataset instacart --number 0 --best_model_path XXX
```
Note, DNNTSP will save several models during the training, an epoch model will be saved if it has higher performance than previous epoch, so XXX is the path of the last model saved during the training.

Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json


### TREx


Get repeat results of TREx, which is saved as 'results/{dataset}_pred0.json', 'results/{dataset}_rel0.json'
```
python repeat.py --dataset instacart --alpha 0.3 --beta 0.8
python repeat.py --dataset dunnhumby --alpha 0.7 --beta 0.9
python repeat.py --dataset tafeng --alpha 0.2 --beta 0.9

```

