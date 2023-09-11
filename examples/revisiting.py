"""
Reported (reproduced) results of FT-Transformer
https://arxiv.org/abs/2106.11959

adult 85.9 (85.5)
helena 39.1 (39.2)
jannis 73.2 (72.2)
california_housing 0.459 (0.537)
--------
Reported (reproduced) results of ResNet
https://arxiv.org/abs/2106.11959

adult 85.7 (85.4)
helena 39.6 (39.1)
jannis 72.8 (72.5)
california_housing 0.486 (0.523)
"""
import argparse
import os.path as osp
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_frame import TaskType, stype
from torch_frame.data import DataLoader
from torch_frame.datasets import Yandex
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearBucketEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    ResNet,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--numerical_encoder_type', type=str, default='linear',
                    choices=['linear', 'linearbucket', 'linearperiodic'])
parser.add_argument('--model_type', type=str, default='fttransformer',
                    choices=['fttransformer', 'resnet'])
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = (torch.device('cuda')
          if torch.cuda.is_available() else torch.device('cpu'))
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = Yandex(root=path, name=args.dataset)
dataset.materialize()
is_classification = (dataset.task_type == TaskType.BINARY_CLASSIFICATION or
                     dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION)

train_dataset = dataset.get_split_dataset('train')
val_dataset = dataset.get_split_dataset('val')
test_dataset = dataset.get_split_dataset('test')

# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame.to(device)
val_tensor_frame = val_dataset.tensor_frame.to(device)
test_tensor_frame = test_dataset.tensor_frame.to(device)
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

if args.numerical_encoder_type == 'linear':
    numerical_encoder = LinearEncoder()
elif args.numerical_encoder_type == 'linearbucket':
    numerical_encoder = LinearBucketEncoder()
elif args.numerical_encoder_type == 'linearperiodic':
    numerical_encoder = LinearPeriodicEncoder()
else:
    raise ValueError(
        f'Unsupported encoder type: {args.numerical_encoder_type}')

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: numerical_encoder,
}

if is_classification:
    output_channels = dataset.num_classes
else:
    output_channels = 1

if args.model_type == 'fttransformer':
    model = FTTransformer(
        channels=args.channels,
        out_channels=output_channels,
        num_layers=args.num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)
elif args.model_type == 'resnet':
    model = ResNet(
        channels=args.channels,
        out_channels=output_channels,
        num_layers=args.num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
    ).to(device)
else:
    raise ValueError(f'Unsupported model type: {args.model_type}')

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()
    loss_accum = 0

    for step, tf in enumerate(tqdm(train_loader)):
        pred = model(tf)
        if is_classification:
            loss = F.cross_entropy(pred, tf.y)
        else:
            loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss)
        optimizer.step()
    return loss_accum / (step + 1)


@torch.no_grad()
def eval(loader: DataLoader) -> Dict[str, float]:
    model.eval()
    accum = 0
    total_count = 0

    for tf in loader:
        pred = model(tf)
        if is_classification:
            pred_class = pred.argmax(dim=-1)
            accum += float((tf.y == pred_class).sum())
        else:
            accum += float(
                F.mse_loss(pred.view(-1), tf.y.view(-1), reduction='sum'))
        total_count += len(tf.y)
    if is_classification:
        accuracy = accum / total_count
        return {"accuracy": accuracy}
    else:
        rmse = (accum / total_count)**0.5
        return {"rmse": rmse}


if is_classification:
    best_val_metric = 0
    best_test_metric = 0
else:
    best_val_metric = float('inf')
    best_test_metric = float('inf')

for epoch in range(args.epochs):
    print(f"=====epoch {epoch}")
    loss = train()
    print(f'Train loss: {loss}')
    train_metrics = eval(train_loader)
    val_metrics = eval(val_loader)
    test_metrics = eval(test_loader)

    if is_classification:
        metric_name = "accuracy"
        if val_metrics[metric_name] > best_val_metric:
            best_val_metric = val_metrics[metric_name]
            best_test_metric = test_metrics[metric_name]
    else:
        metric_name = "rmse"
        if val_metrics[metric_name] < best_val_metric:
            best_val_metric = val_metrics[metric_name]
            best_test_metric = test_metrics[metric_name]

    print(f'Train {metric_name}: {train_metrics[metric_name]}, '
          f'val {metric_name}: {val_metrics[metric_name]}, '
          f'test {metric_name}: {test_metrics[metric_name]}')

print(f'Best val {metric_name}: {best_val_metric}, '
      f'best test {metric_name}: {best_test_metric}')
