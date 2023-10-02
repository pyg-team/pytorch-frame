"""
Reported (reproduced) accuracy(rmse for regression task) of ExcelFormer based
on Table 1 of the paper. https://arxiv.org/pdf/2301.02819.pdf
ExcelFormer uses the same train-validation-test split as the Yandex paper.

california: 0.4587 (0.4733) num_layers=5, num_heads=4, num_layers=5,
channels=32, lr: 0.001,
jannis : 72.51 (72.38) num_heads=32, lr: 0.0001
covtype: 97.17 (95.37)
helena: 38.20 (36.80)
higgs_small: 80.75 (65.17) lr: 0.0001
"""
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame.data.loader import DataLoader
from torch_frame.datasets.yandex import Yandex
from torch_frame.nn import ExcelFormer
from torch_frame.transforms import (
    MutualInformationSort,
    OrderedTargetStatisticsEncoder,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='higgs_small')
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--mixup', type=bool, default=True)
parser.add_argument('--beta', type=float, default=0.5)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = Yandex(root=path, name=args.dataset)
dataset.materialize()
train_dataset = dataset.get_split_dataset('train')
val_dataset = dataset.get_split_dataset('val')
test_dataset = dataset.get_split_dataset('test')
train_tensor_frame = train_dataset.tensor_frame.to(device)
val_tensor_frame = val_dataset.tensor_frame.to(device)
test_tensor_frame = test_dataset.tensor_frame.to(device)

# CategoricalCatBoostEncoder encodes the categorical features
# into numerical features with CatBoostEncoder.
categorical_transform = OrderedTargetStatisticsEncoder()
categorical_transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)

train_tensor_frame = categorical_transform(train_tensor_frame)
val_tensor_frame = categorical_transform(val_tensor_frame)
test_tensor_frame = categorical_transform(test_tensor_frame)
col_stats = categorical_transform.transformed_stats

# MutualInformationSort sorts the features based on mutual
# information.
mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)

mutual_info_sort.fit(train_tensor_frame, col_stats)
train_tensor_frame = mutual_info_sort(train_tensor_frame)
val_tensor_frame = mutual_info_sort(val_tensor_frame)
test_tensor_frame = mutual_info_sort(test_tensor_frame)

train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

is_classification = dataset.task_type.is_classification

if is_classification:
    out_channels = dataset.num_classes
else:
    out_channels = 1

model = ExcelFormer(
    in_channels=args.channels,
    out_channels=out_channels,
    num_layers=args.num_layers,
    num_cols=len(dataset.col_to_stype) - 1,
    num_heads=args.num_heads,
    residual_dropout=0.,
    diam_dropout=0.3,
    aium_dropout=0.,
    col_stats=mutual_info_sort.transformed_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        pred_mixedup, y_mixedup = model.forward_mixup(tf)
        if is_classification:
            loss = F.cross_entropy(pred_mixedup, y_mixedup)
        else:
            loss = F.mse_loss(pred_mixedup.view(-1), y_mixedup.view(-1))
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(y_mixedup)
        total_count += len(y_mixedup)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    accum = total_count = 0

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
        return accuracy
    else:
        rmse = (accum / total_count)**0.5
        return rmse


if is_classification:
    metric = 'Acc'
    best_val_metric = 0
    best_test_metric = 0
else:
    metric = 'RMSE'
    best_val_metric = float('inf')
    best_test_metric = float('inf')

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)

    if is_classification and val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
    elif not is_classification and val_metric < best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
          f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')

print(f'Best Val {metric}: {best_val_metric:.4f}, '
      f'Best Test {metric}: {best_test_metric:.4f}')
