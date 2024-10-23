# Source Code and Appendix for "Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation"

This paper proposes a model-agnostic repeat-bias-aware diversity optimization (RADiv) algorithm and a repeat-bias-aware item fairness optimization (RAIF) algorithm based on ILP for next basket recommendation (NBR).

We apply the proposed algorithms on 5 representative NBR methods to mitigate repeat bias while improving the diversity and item fairness of recommended baskets.
 


## Required packages

To run our rerank scripts, gurobipy, Numpy and Python >= 3.10 are required.

To run the published NBR methods' code, please go to the original repository and check the required packages.

## Dataset

* Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis/data
* Dunnhumby: https://www.dunnhumby.com/source-files/
* Tafeng: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset

We provide the scripts of preprocessing, and the preprocessed dataset with different formats (csvdata, jsondata, mergedata), which can be used directly.

### Format description of preprocessed dataset
* csvdata: --> TREx
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
* ra_rerank: RADiv and RAIF algorithms.
* evaluation: scripts for evaluation.
    * fair_metrics: the fairness metrics.
    * diversity_metrics.py: the diversity metrics. 
    * metrics.py: the accuracy metrics.
    * model_performance.py: evaluate the fairness, diversity and accuracy of recommendation results.
* methods: contains 5 NBR methods.
* appendix: contains a PDF file with the additional experiment results.


## Pipeline

* Step 1. Run NBR methods and save the results. (Note that we use the original implementations of the authors, so we provide the original repository links, which contain the instructions of the environment setting, how to run each method, etc. We also provide our additional instructions and hyperparameters in the following section, which can make the running easier.)


* Step 2. Re-rank: apply RADiv and RAIF to re-rank the recommendation obtained from NBR methods.


* Step 3. Evaluate: Use the evaluation scripts to get the performance before and after re-ranking.


## Rerank

For unified NBR methods, such as UP-CF@r, TIFUKNN, Dream, DNNTSP

```
python rerank_overall_bias.py --topk 100 --size 20 --method upcf --pred_folder XXX

```
For combined NBR methods, such as TREx:
```
python rerank_trex_test.py --topk 100 --size 20 --method_rep trep --method_expl upcf --dataset dunnhumby --pred_folder_rep XXX --pred_folder_expl XXX --theta_list XXX

```

### Hyperparameters for each NBR method:

RADiv:
|            | Instacart               | Dunnhumby               | TaFeng                  |
|------------|------------|------------|------------|------------|------------|------------|
|            | epsilon_1  | lambda     |epsilon_1   | lambda     |epsilon_1   | lambda     |
|------------|------------|------------|------------|------------|------------|------------|
| UP-CF@r    |   0.2      |   0.01     |  0.08      |   0.01     |   0.08     |   0.001    |
| TIFUKNN    |   0.2      |  0.01      |  0.04      |  0.001     |   0.01     |  0.001     |
| Dream      |   0.18     |   0.3      |  0.12      |   0.01     |   0.1      |    0.1     |
| DNNTSP     |    0.2     |    0.01    |   0.12     |    0.01    |   0.01     |    0.01    |
|            | epsilon_2  | theta      |epsilon_2   | theta      |epsilon_2   | theta      |
| TREx       |    0.5     |    0.34249 |   10       |    0.57303 |   0.001    |    0.12545 |







RAIF:
|            | Instacart               | Dunnhumby               | TaFeng                  |
|            |------------|------------|------------|------------|------------|------------|
|            | alpha_1    | lambda     |alpha_1     | lambda     |alpha_1     | lambda     |
|------------|------------|------------|------------|------------|------------|------------|
| UP-CF@r    |   200      |   0.9      |  50        |   0.3      |   100      |   0.8      |
| TIFUKNN    |   200      |  0.8       |  1         |  0.001     |   20       |  0.1       |
| Dream      |   200      |   0.3      |  200       |   0.5      |   20       |    0.2     |
| DNNTSP     |   200      |    0.9     |   200      |    0.4     |   200      |    0.6     |
|------------|------------|------------|------------|------------|------------|------------|
|            | alpha_2    | theta      |alpha_2     | theta      |alpha_2     | theta      |
|------------|------------|------------|------------|------------|------------|------------|
| TREx       |    200     |    0.4039  |  200       |    0.62322 |   200      |    0.0501  |

## evaluation 

```

python evaluate_overall_bias.py --pred_folder XXX --fold_list 0 --eval XXX --item_eps_list 0 0.001 0.01 0.1 1 10 20 30 40 50 60 70 80 90 100 200 --lamda_list 0 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 --method dnntsp --dataset tafeng

```








## Guidelines for NBR methods




