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



## evaluation 



## Guidelines for NBR methods




