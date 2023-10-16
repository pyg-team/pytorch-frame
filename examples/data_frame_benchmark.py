import argparse
import math
import os
import os.path as osp
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.gbdt import CatBoost, XGBoost
from torch_frame.nn.encoder import EmbeddingEncoder, LinearBucketEncoder
from torch_frame.nn.models import (
    ExcelFormer,
    FTTransformer,
    ResNet,
    TabNet,
    TabTransformer,
    Trompt,
)
from torch_frame.typing import TaskType

TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]
GBDT_MODELS = ["XGBoost", "CatBoost"]

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
parser.add_argument(
    '--model_type', type=str, default='TabNet', choices=[
        'TabNet', 'FTTransformer', 'ResNet', 'TabTransformer', 'Trompt',
        'ExcelFormer', 'FTTransformerBucket', 'XGBoost', 'CatBoost'
    ])
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
train_dataset = dataset.get_split_dataset('train')
val_dataset = dataset.get_split_dataset('val')
test_dataset = dataset.get_split_dataset('test')

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame

if args.model_type in GBDT_MODELS:
    gbdt_cls_dict = {'XGBoost': XGBoost, 'CatBoost': CatBoost}
    model_cls = gbdt_cls_dict[args.model_type]
else:
    if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fun = BCEWithLogitsLoss()
        metric_computer = AUROC(task='binary').to(device)
        higher_is_better = True
    elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = dataset.num_classes
        loss_fun = CrossEntropyLoss()
        metric_computer = Accuracy(task='multiclass',
                                   num_classes=dataset.num_classes).to(device)
        higher_is_better = True
    elif dataset.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fun = MSELoss()
        metric_computer = MeanSquaredError(squared=False).to(device)
        higher_is_better = False

    # To be set for each model
    model_cls = None
    col_stats = None

    # Set up model specific search space
    if args.model_type == 'TabNet':
        model_search_space = {
            'split_attention_channels': [64, 128, 256],
            'split_feature_channels': [64, 128, 256],
            'gamma': [1., 1.2, 1.5],
            'num_layers': [4, 6, 8],
        }
        train_search_space = {
            'batch_size': [2048, 4096],
            'base_lr': [0.001, 0.01],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = TabNet
        col_stats = dataset.col_stats
    elif args.model_type == 'FTTransformer':
        model_search_space = {
            'channels': [64, 128, 256],
            'num_layers': [4, 6, 8],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = FTTransformer
        col_stats = dataset.col_stats
    elif args.model_type == 'FTTransformerBucket':
        model_search_space = {
            'channels': [64, 128, 256],
            'num_layers': [4, 6, 8],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = FTTransformer

        col_stats = dataset.col_stats
    elif args.model_type == 'ResNet':
        model_search_space = {
            'channels': [64, 128, 256],
            'num_layers': [4, 6, 8],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = ResNet
        col_stats = dataset.col_stats
    elif args.model_type == 'TabTransformer':
        model_search_space = {
            'channels': [16, 32, 64, 128],
            'num_layers': [4, 6, 8],
            'num_heads': [4, 8],
            'encoder_pad_size': [2, 4],
            'attn_dropout': [0, 0.2],
            'ffn_dropout': [0, 0.2],
        }
        train_search_space = {
            'batch_size': [128, 256],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = TabTransformer
        col_stats = dataset.col_stats
    elif args.model_type == 'Trompt':
        model_search_space = {
            'channels': [64, 128, 192],
            'num_layers': [4, 6, 8],
            'num_prompts': [64, 128, 192],
        }
        train_search_space = {
            'batch_size': [128, 256],
            'base_lr': [0.01, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        if train_tensor_frame.num_cols > 20:
            # Reducing the model size to avoid GPU OOM
            model_search_space['channels'] = [64, 128]
            model_search_space['num_prompts'] = [64, 128]
        model_cls = Trompt
        col_stats = dataset.col_stats
    elif args.model_type == 'ExcelFormer':
        from torch_frame.transforms import (
            CatToNumTransform,
            MutualInformationSort,
        )

        categorical_transform = CatToNumTransform()
        categorical_transform.fit(train_dataset.tensor_frame,
                                  train_dataset.col_stats)
        train_tensor_frame = categorical_transform(train_tensor_frame)
        val_tensor_frame = categorical_transform(val_tensor_frame)
        test_tensor_frame = categorical_transform(test_tensor_frame)
        col_stats = categorical_transform.transformed_stats

        mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)
        mutual_info_sort.fit(train_tensor_frame, col_stats)
        train_tensor_frame = mutual_info_sort(train_tensor_frame)
        val_tensor_frame = mutual_info_sort(val_tensor_frame)
        test_tensor_frame = mutual_info_sort(test_tensor_frame)

        model_search_space = {
            'in_channels': [128, 256],
            'num_heads': [8, 16, 32],
            'num_layers': [4, 6, 8],
            'diam_dropout': [0, 0.2],
            'residual_dropout': [0, 0.2],
            'aium_dropout': [0, 0.2],
            'num_cols': [train_tensor_frame.num_cols],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = ExcelFormer

    assert model_cls is not None
    assert col_stats is not None
    assert set(train_search_space.keys()) == set(TRAIN_CONFIG_KEYS)
    col_names_dict = train_tensor_frame.col_names_dict


def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        y = tf.y
        if isinstance(model, ExcelFormer):
            pred, y = model.forward_mixup(tf)
        else:
            pred = model.forward(tf)
            if isinstance(model, Trompt):
                # Trompt uses the layer-wise loss
                num_layers = pred.size(1)
                # [batch_size * num_layers, num_classes]
                pred = pred.view(-1, out_channels)
                y = tf.y.repeat_interleave(num_layers)

        if pred.size(1) == 1:
            pred = pred.view(-1, )
        if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float)
        loss = loss_fun(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(
    model: Module,
    loader: DataLoader,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if isinstance(model, Trompt):
            pred = pred.mean(dim=1)  # [batch_size, out_channels]
        if dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif dataset.task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()


def train_and_eval_with_cfg(
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    trial: Optional[optuna.trial.Trial] = None,
) -> Tuple[float, float]:
    # Use model_cfg to set up training procedure
    if args.model_type == 'FTTransformerBucket':
        # Use LinearBucketEncoder instead
        stype_encoder_dict = {
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearBucketEncoder(),
        }
        model_cfg['stype_encoder_dict'] = stype_encoder_dict
    model = model_cls(
        **model_cfg,
        out_channels=out_channels,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
    ).to(device)
    model.reset_parameters()
    # Use train_cfg to set up training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['base_lr'])
    lr_scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_rate'])
    train_loader = DataLoader(train_tensor_frame,
                              batch_size=train_cfg['batch_size'], shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_tensor_frame,
                            batch_size=train_cfg['batch_size'])
    test_loader = DataLoader(test_tensor_frame,
                             batch_size=train_cfg['batch_size'])

    if higher_is_better:
        best_val_metric = 0
    else:
        best_val_metric = math.inf

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        val_metric = test(model, val_loader)

        if higher_is_better:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader)
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader)
        lr_scheduler.step()
        print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

        if trial is not None:
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    print(
        f'Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}')
    return best_val_metric, best_test_metric


def objective(trial: optuna.trial.Trial) -> float:
    model_cfg = {}
    for name, search_list in model_search_space.items():
        model_cfg[name] = trial.suggest_categorical(name, search_list)
    train_cfg = {}
    for name, search_list in train_search_space.items():
        train_cfg[name] = trial.suggest_categorical(name, search_list)

    best_val_metric, _ = train_and_eval_with_cfg(model_cfg=model_cfg,
                                                 train_cfg=train_cfg,
                                                 trial=trial)
    return best_val_metric


def main_deep_models():
    # Hyper-parameter optimization with Optuna
    print("Hyper-parameter search via Optuna")
    start_time = time.time()
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize" if higher_is_better else "minimize",
    )
    study.optimize(objective, n_trials=args.num_trials)
    end_time = time.time()
    search_time = end_time - start_time
    print("Hyper-parameter search done. Found the best config.")
    params = study.best_params
    best_train_cfg = {}
    for train_cfg_key in TRAIN_CONFIG_KEYS:
        best_train_cfg[train_cfg_key] = params.pop(train_cfg_key)
    best_model_cfg = params

    print(f"Repeat experiments {args.num_repeats} times with the best train "
          f"config {best_train_cfg} and model config {best_model_cfg}.")
    start_time = time.time()
    best_val_metrics = []
    best_test_metrics = []
    for _ in range(args.num_repeats):
        best_val_metric, best_test_metric = train_and_eval_with_cfg(
            best_model_cfg, best_train_cfg)
        best_val_metrics.append(best_val_metric)
        best_test_metrics.append(best_test_metric)
    end_time = time.time()
    final_model_time = (end_time - start_time) / args.num_repeats
    best_val_metrics = np.array(best_val_metrics)
    best_test_metrics = np.array(best_test_metrics)

    result_dict = {
        'args': args.__dict__,
        'best_val_metrics': best_val_metrics,
        'best_test_metrics': best_test_metrics,
        'best_val_metric': best_val_metrics.mean(),
        'best_test_metric': best_test_metrics.mean(),
        'best_train_cfg': best_train_cfg,
        'best_model_cfg': best_model_cfg,
        'search_time': search_time,
        'final_model_time': final_model_time,
        'total_time': search_time + final_model_time,
    }
    print(result_dict)
    # Save results
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_dict, args.result_path)


def main_gbdt():
    if dataset.task_type.is_classification:
        num_classes = dataset.num_classes
    else:
        num_classes = None
    model = model_cls(task_type=dataset.task_type, num_classes=num_classes)

    import time
    start_time = time.time()
    model.tune(tf_train=train_dataset.tensor_frame,
               tf_val=val_dataset.tensor_frame, num_trials=args.num_trials)
    val_pred = model.predict(tf_test=val_dataset.tensor_frame)
    val_metric = model.compute_metric(val_dataset.tensor_frame.y,
                                      val_pred)[model.metric]
    test_pred = model.predict(tf_test=test_dataset.tensor_frame)
    test_metric = model.compute_metric(test_dataset.tensor_frame.y,
                                       test_pred)[model.metric]
    end_time = time.time()
    result_dict = {
        'args': args.__dict__,
        'best_val_metric': val_metric,
        'best_test_metric': test_metric,
        'best_cfg': model.params,
        'total_time': end_time - start_time,
    }
    print(result_dict)
    # Save results
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_dict, args.result_path)


if __name__ == '__main__':
    print(args)
    if os.path.exists(args.result_path):
        exit(-1)
    if args.model_type in ["XGBoost", "CatBoost"]:
        main_gbdt()
    else:
        main_deep_models()
