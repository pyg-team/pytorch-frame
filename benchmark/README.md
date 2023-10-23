## Benchmarking model performance across diverse `DataFrameBenchmark` datasets.

First install additional dependencies:
```bash
pip install optuna
pip install torchmetrics
pip install xgboost
pip install catboost
```

Then run
```bash
# Specify the model from [TabNet, FTTransformer, ResNet, TabTransformer, Trompt
# ExcelFormer, FTTransformerBucket, XGBoost, CatBoost]
model_type=TabNet

# Specify the task type from [binary_classification, regression,
# multiclass_classification]
task_type=binary_classification

# Specify the dataset scale from [small, medium, large]
scale=small

# Specify the dataset idx from [0, 1, ...]
idx=0

# Specify the path to save the results
result_path=results.pt

# Run hyper-parameter tuning and training of the specified model on a specified
# dataset.
python data_frame_benchmark.py --model_type $model_type --task_type $task_type --scale $scale --idx $idx --result_path $result_path
```

## Leaderboard

We show the current model performance across different datasets.
The row denotes the model names and the column denotes the dataset `idx`.
In each cell, we include the mean and standard deviation of the model performance, as well as
the total time spent, including [`Optuna`](https://optuna.org/)-based hyper-parameter search and final model training.

For the mapping from dataset `idx` into the actual dataset object, please see the [documentation](https://pytorch-frame.readthedocs.io/en/latest/generated/torch_frame.datasets.DataFrameBenchmark.html#torch_frame.datasets.DataFrameBenchmark).

### `task_type: binary_classification`
Metric: ROC-AUC, higher the better.

#### `scale: small`

Experimental setting: 20 Optuna search trials. 50 epochs of training.

|                     | dataset_0                      | dataset_1                       | dataset_2                        | dataset_3                     | dataset_4                       | dataset_5                      | dataset_6                     | dataset_7                      | dataset_8                      | dataset_9                      | dataset_10                    | dataset_11                      | dataset_12                      | dataset_13                      |
|:--------------------|:-----------------------|:------------------------|:-------------------------|:----------------------|:------------------------|:-----------------------|:----------------------|:-----------------------|:-----------------------|:-----------------------|:----------------------|:------------------------|:------------------------|:------------------------|
| XGBoost             | **0.931±0.000 (41s)**  | **1.000±0.000 (4s)**    | 0.935±0.000 (16s)        | **0.946±0.000 (26s)** | **0.881±0.000 (10s)**   | 0.951±0.000 (16s)      | **0.862±0.000 (26s)** | **0.780±0.000 (11s)**  | **0.983±0.000 (584s)** | **0.763±0.000 (240s)** | **0.795±0.000 (11s)** | 0.950±0.000 (479s)      | **0.999±0.000 (148s)**  | **0.926±0.000 (3042s)** |
| CatBoost            | **0.930±0.000 (152s)** | **1.000±0.000 (9s)**    | 0.938±0.000 (164s)       | 0.924±0.000 (29s)     | **0.881±0.000 (27s)**   | 0.963±0.000 (48s)      | **0.861±0.000 (12s)** | 0.772±0.000 (10s)      | 0.930±0.000 (91s)      | 0.628±0.000 (10s)      | **0.796±0.000 (15s)** | 0.948±0.000 (46s)       | **0.998±0.000 (38s)**   | **0.926±0.000 (115s)**  |
| Trompt              | 0.919±0.000 (9627s)    | **1.000±0.000 (5341s)** | **0.945±0.000 (14679s)** | 0.942±0.001 (2752s)   | **0.881±0.001 (2640s)** | 0.964±0.001 (5173s)    | 0.855±0.002 (4249s)   | 0.778±0.002 (8789s)    | 0.933±0.001 (9353s)    | 0.686±0.008 (3105s)    | 0.793±0.002 (8255s)   | **0.952±0.001 (4876s)** | **1.000±0.000 (3558s)** | 0.916±0.001 (30002s)    |
| ResNet              | 0.917±0.000 (615s)     | **1.000±0.000 (71s)**   | 0.937±0.001 (787s)       | 0.938±0.002 (230s)    | 0.865±0.001 (183s)      | 0.960±0.001 (349s)     | 0.828±0.001 (248s)    | 0.768±0.002 (205s)     | 0.925±0.002 (958s)     | 0.665±0.006 (140s)     | **0.794±0.002 (76s)** | 0.946±0.002 (145s)      | **1.000±0.000 (93s)**   | 0.911±0.001 (880s)      |
| FTTransformerBucket | 0.915±0.001 (690s)     | **0.999±0.001 (354s)**  | 0.936±0.002 (1705s)      | 0.939±0.002 (484s)    | 0.876±0.002 (321s)      | 0.960±0.001 (746s)     | 0.857±0.000 (549s)    | 0.771±0.003 (654s)     | 0.909±0.002 (1177s)    | 0.636±0.012 (244s)     | 0.788±0.002 (710s)    | 0.950±0.001 (510s)      | **0.999±0.000 (634s)**  | 0.913±0.001 (1164s)     |
| ExcelFormer         | 0.918±0.001 (1587s)    | **1.000±0.000 (634s)**  | 0.939±0.001 (1827s)      | 0.939±0.002 (378s)    | 0.878±0.003 (251s)      | **0.969±0.000 (678s)** | 0.833±0.011 (435s)    | **0.780±0.002 (938s)** | 0.921±0.005 (1131s)    | 0.649±0.008 (519s)     | 0.794±0.003 (683s)    | 0.950±0.001 (405s)      | **0.999±0.000 (1169s)** | 0.919±0.001 (1798s)     |
| FTTransformer       | 0.918±0.001 (871s)     | **1.000±0.000 (571s)**  | 0.940±0.001 (1371s)      | 0.936±0.001 (458s)    | 0.874±0.002 (200s)      | 0.959±0.001 (622s)     | 0.828±0.001 (339s)    | 0.773±0.002 (521s)     | 0.909±0.002 (1488s)    | 0.635±0.011 (392s)     | 0.790±0.001 (556s)    | 0.949±0.002 (374s)      | **1.000±0.000 (713s)**  | 0.912±0.000 (1855s)     |
| TabNet              | 0.911±0.001 (150s)     | **1.000±0.000 (35s)**   | 0.931±0.005 (254s)       | 0.937±0.003 (125s)    | 0.864±0.002 (52s)       | 0.944±0.001 (116s)     | 0.828±0.001 (79s)     | 0.771±0.005 (93s)      | 0.913±0.005 (177s)     | 0.606±0.014 (65s)      | 0.790±0.003 (41s)     | 0.936±0.003 (104s)      | **1.000±0.000 (64s)**   | 0.910±0.001 (294s)      |
| TabTransformer      | 0.910±0.001 (2044s)    | **1.000±0.000 (1321s)** | 0.928±0.001 (2519s)      | 0.918±0.003 (134s)    | 0.829±0.002 (64s)       | 0.928±0.001 (105s)     | 0.816±0.002 (99s)     | 0.757±0.003 (645s)     | 0.885±0.001 (1167s)    | 0.652±0.006 (282s)     | 0.780±0.002 (112s)    | 0.937±0.001 (117s)      | 0.996±0.000 (76s)       | 0.905±0.001 (2283s)     |

#### `scale: medium`

TODO: Add results.

Experimental setting: 10 Optuna search trials. 25 epochs of training.

#### `scale: large`

TODO: Add results.

### `task_type: regression`
Metric: RMSE, lower the better.

#### `scale: small`

Experimental setting: 20 Optuna search trials. 50 epochs of training.

|                     | dataset_0                      | dataset_1                       | dataset_2                       | dataset_3                        | dataset_4                       | dataset_5                      | dataset_6                      | dataset_7                      | dataset_8                       | dataset_9                       | dataset_10                     | dataset_11                   | dataset_12                      |
|:--------------------|:-----------------------|:------------------------|:------------------------|:-------------------------|:------------------------|:-----------------------|:-----------------------|:-----------------------|:------------------------|:------------------------|:-----------------------|:---------------------|:------------------------|
| XGBoost             | **0.247±0.000 (516s)** | 0.077±0.000 (14s)       | 0.167±0.000 (423s)      | 1.119±0.000 (1063s)      | 0.328±0.000 (2044s)     | 1.024±0.000 (47s)      | **0.292±0.000 (844s)** | 0.606±0.000 (1765s)    | **0.876±0.000 (2288s)** | 0.023±0.000 (1170s)     | **0.697±0.000 (248s)** | **0.865±0.000 (8s)** | 0.435±0.000 (22s)       |
| CatBoost            | 0.265±0.000 (116s)     | 0.062±0.000 (129s)      | 0.128±0.000 (97s)       | 0.336±0.000 (103s)       | 0.346±0.000 (110s)      | 0.443±0.000 (97s)      | 0.375±0.000 (46s)      | **0.273±0.000 (693s)** | 0.881±0.000 (660s)      | 0.040±0.000 (80s)       | 0.756±0.000 (44s)      | 0.876±0.000 (110s)   | 0.439±0.000 (101s)      |
| Trompt              | 0.261±0.003 (8390s)    | **0.015±0.005 (3792s)** | **0.118±0.001 (3836s)** | **0.262±0.001 (10037s)** | **0.323±0.001 (9255s)** | 0.418±0.003 (9071s)    | 0.329±0.009 (2977s)    | 0.312±0.002 (21967s)   | OOM                     | **0.008±0.001 (1889s)** | 0.779±0.006 (775s)     | 0.874±0.004 (3723s)  | **0.424±0.005 (3185s)** |
| ResNet              | 0.288±0.006 (220s)     | 0.018±0.003 (187s)      | 0.124±0.001 (135s)      | 0.268±0.001 (330s)       | 0.335±0.001 (471s)      | 0.434±0.004 (345s)     | 0.325±0.012 (178s)     | 0.324±0.004 (365s)     | 0.895±0.005 (142s)      | 0.036±0.002 (172s)      | 0.794±0.006 (120s)     | 0.875±0.004 (122s)   | 0.468±0.004 (303s)      |
| FTTransformerBucket | 0.325±0.008 (619s)     | 0.096±0.005 (290s)      | 0.360±0.354 (332s)      | 0.284±0.005 (768s)       | 0.342±0.004 (757s)      | 0.441±0.003 (835s)     | 0.345±0.007 (191s)     | 0.339±0.003 (3321s)    | OOM                     | 0.105±0.011 (199s)      | 0.807±0.010 (156s)     | 0.885±0.008 (820s)   | 0.468±0.006 (706s)      |
| ExcelFormer         | 0.302±0.003 (703s)     | 0.099±0.003 (490s)      | 0.145±0.003 (587s)      | 0.382±0.011 (504s)       | 0.344±0.002 (1096s)     | **0.411±0.005 (469s)** | 0.359±0.016 (207s)     | 0.336±0.008 (5522s)    | OOM                     | 0.192±0.014 (317s)      | 0.794±0.005 (189s)     | 0.890±0.003 (1186s)  | 0.445±0.005 (550s)      |
| FTTransformer       | 0.335±0.010 (338s)     | 0.161±0.022 (370s)      | 0.140±0.002 (244s)      | 0.277±0.004 (516s)       | 0.335±0.003 (973s)      | 0.445±0.003 (599s)     | 0.361±0.018 (286s)     | 0.345±0.005 (2443s)    | OOM                     | 0.106±0.012 (150s)      | 0.826±0.005 (121s)     | 0.896±0.007 (832s)   | 0.461±0.003 (647s)      |
| TabNet              | 0.279±0.003 (68s)      | 0.224±0.016 (53s)       | 0.141±0.010 (34s)       | 0.275±0.002 (61s)        | 0.348±0.003 (110s)      | 0.451±0.007 (82s)      | 0.355±0.030 (49s)      | 0.332±0.004 (168s)     | 0.992±0.182 (53s)       | 0.015±0.002 (57s)       | 0.805±0.014 (27s)      | 0.885±0.013 (46s)    | 0.544±0.011 (112s)      |
| TabTransformer      | 0.624±0.003 (1225s)    | 0.229±0.003 (1200s)     | 0.369±0.005 (52s)       | 0.340±0.004 (163s)       | 0.388±0.002 (1137s)     | 0.539±0.003 (100s)     | 0.619±0.005 (73s)      | 0.351±0.001 (125s)     | 0.893±0.005 (389s)      | 0.431±0.001 (489s)      | 0.819±0.002 (52s)      | 0.886±0.005 (46s)    | 0.545±0.004 (95s)       |

#### `scale: medium`

TODO: Add results.

Experimental setting: 10 Optuna search trials. 25 epochs of training.

#### `scale: large`

TODO: Add results.

### `task_type: multiclass_classification`
Metric: Accuracy, the higher the better.


#### `scale: medium`

TODO: Add results.

Experimental setting: 10 Optuna search trials. 25 epochs of training.

#### `scale: large`

TODO
