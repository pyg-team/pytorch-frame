import argparse
import os.path as osp

import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from sklearn.metrics import roc_auc_score

from torch_frame import TaskType, stype
from torch_frame.datasets import DataFrameBenchmark


def roc_auc(y_hat, y):
    r"""Calculate the Area Under the ROC Curve (AUC)
    for the given predictions and true labels.

    Parameters:
    y_hat (array-like): Predicted probabilities or scores.
    y (array-like): True binary labels.

    Returns:
    float: AUC score.
    """
    return roc_auc_score(y, y_hat)


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

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                             scale=args.scale, idx=args.idx)
dataset.materialize()
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()

train_df, val_df, test_df, target_col, cat_col_names, num_col_names = (
    train_dataset.df, val_dataset.df, test_dataset.df, dataset.target_col,
    dataset.tensor_frame.col_names_dict[stype.categorical],
    dataset.tensor_frame.col_names_dict[stype.numerical])

data_config = DataConfig(
    target=[target_col],
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
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
tabular_model.fit(train=train_df, validation=val_df, metrics=roc_auc)
tabular_model.evaluate(test_df)
