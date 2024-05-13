import argparse
import os.path as osp

import numpy as np
import pandas as pd
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.models.tab_transformer import TabTransformerConfig

from torch_frame import TaskType, stype
from torch_frame.datasets import DataFrameBenchmark

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type', type=str, choices=[
        'binary_classification',
        'multiclass_classification',
        'regression',
    ], default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                    default='small')
parser.add_argument('--idx', type=int, default=0,
                    help='The index of the dataset within DataFrameBenchmark')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--num_trials', type=int, default=20,
                    help='Number of Optuna-based hyper-parameter tuning.')
parser.add_argument(
    '--num_repeats', type=int, default=5,
    help='Number of repeated training and eval on the best config.')
parser.add_argument('--model_type', type=str, default='TabNet',
                    choices=['TabNet', 'FTTransformer', 'TabTransformer'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--result_path', type=str, default='')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)


def load_classification_data(df, target_col, test_size):
    torch_data = np.array(df.drop(target_col, axis=1))
    torch_labels = np.array(df[target_col])
    data = np.hstack([torch_data, torch_labels.reshape(-1, 1)])
    gen_names = [f"feature_{i}" for i in range(data.shape[-1])]
    col_names = gen_names
    col_names[-1] = "target"
    data = pd.DataFrame(data, columns=col_names)
    cat_col_names = [x for x in gen_names[:-1] if len(data[x].unique()) < 10]
    num_col_names = [
        x for x in gen_names[:-1] if x not in [target_col] + cat_col_names
    ]
    test_idx = data.sample(int(test_size * len(data)), random_state=42).index
    test = data[data.index.isin(test_idx)]
    train = data[~data.index.isin(test_idx)]
    return (train, test, ["target"], cat_col_names, num_col_names)


# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                             scale=args.scale, idx=args.idx)
dataset.materialize()
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()

train, test, target_col, cat_col_names, num_col_names = (
    train_dataset.df, test_dataset.df, dataset.target_col,
    dataset.tensor_frame.col_names_dict[stype.categorical],
    dataset.tensor_frame.col_names_dict[stype.numerical])

data_config = DataConfig(
    target=[target_col],
    continuous_cols=cat_col_names,
    categorical_cols=num_col_names,
)

trainer_config = TrainerConfig(
    auto_lr_find=True,
    batch_size=256,
    max_epochs=10,
    checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
    load_best=True,  # After training, load the best checkpoint
)

optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="", dropout=0.1, initialization="kaiming"
).__dict__  # Convert to dict to pass to the model config

model_config = TabTransformerConfig(
    task="classification",
    learning_rate=1e-3,
    head="LinearHead",  # Linear Head
    head_config=head_config,  # Linear Head Config
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train)
tabular_model.evaluate(test)
