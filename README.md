Project: TENSOR-BASED INDIVIDUALIZED TREATMENT RULE (TITR)
========================================================

1. Overview

------------------------------------

We present implementation for estimation of TITR model. For low-dimensional setting, the estimate is obtained by an alternating update algorithm based on Tucker decomposition. For sparse scenario, estimation is performed using Tucker decomposition with potentially distinct ranks for treatment-free and treatment-related effects, combined with L1-regularization to induce sparsity, and an ADMM-based optimization algorithm.


2. Simulation

------------------------------------

-   `low_d.py`: Functions for low-dimensional setting.
-   `sparse_testing.py`: Functions for sparse scenario. (bootstrap code)
-   `README.txt`: Description of the two code files above.

The results of Simulation 4.1 (Low-dimensional TITR) can be reproduced by running

```         
TITR_code/Simu/low_d.py
```

This code includes the following functions: 

-   `tensor_r_tucker()`: tensor
regression 
-   `generate_data()`: generate data 
-   `pi_opt()`, `pi_hat()`, `pi_random()`, `pi_notensor()`: Return average
outcome value and corresponding MSEs under different treatment rules.
-   `select_rank()`: select rank for tensor regression by BIC
-   `replication()`: replication of the simulation 

To reproduce Table 1 and Table 2 under different parameter settings, 
modify the arguments in `replication()`: 

-   `n0` sample size, `R0` true rank, `p` tensor dimension, `sigma` noise level.


The results of Simulation 4.2 (Sparse TITR) and 
4.3 (Effectiveness of Inference Procedure) can be reproduced by running

```         
TITR_code/Simu/sparse_testing.py
```

This code includes the following functions: 

-   `ADMM()`: ADMM algorithm to compute factor matrix
-   `tensor_r_lasso()`: tensor regression 
-   `generate_data()`: generate true coefficients when dimension = `(50,50,2)`
-   `generate_sample()`: generate data sample for given coefficients 
-   `pi_opt()`, `pi_hat()`, `pi_random()`, `pi_notensor()`: Return average outcome 
value and corresponding MSEs under different treatment rules.
-   `rankgrid()`: grid search for different ranks, return corresponding estimation 
errors and estimated coefficients.
-   `select_r()`: select rank for tensor regression by loss in test set.
-   `replication()`: replication of the simulation (change argument `n0`)


We use the wild bootstrap to test whether two
treatment-related coefficients are equal to zero. 

-   `bootstrap_value()`: calculate bootstrap statistics
-   `generate_data2()`, `generate_sample2()`: generate coefficients and data
sample under dimension `(10,10,10)`, rank `(2,2,2)` and sigma level `0.5`.
-   `replication()`: calculate p-values under different delta. (change argument `delta`)


3. ADNI Real Data

------------------------------------

For the ADNI real data analysis, please download the dataset from the following link:
[Google Drive – ADNI Data](https://drive.google.com/drive/folders/1HPwFcV81CQaSXvy0Z6HpuKaOk5nub6gM?usp=drive_link)

After downloading, place the data in the following directory:

```
TITR_code/Real Data/ADNI haimati image
```

Then, update the data path accordingly in the notebooks
`left_image_modeling.ipynb` and `right_image_modeling.ipynb`.

------------------------------------


