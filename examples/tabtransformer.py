"""
Reported (reproduced) results of of TabTransformer model based on Table 1
of original paper https://arxiv.org/abs/2012.06678
albert: 75.7 (63.92)
adult: 73.8 (76.05) batch_size: 128, lr: 0.0001, num_heads: 32, num_layers: 6
bank-marketing: 93.4 (76.84)
dota2: 63.3 (58.28)
"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame.data import DataLoader
from torch_frame.datasets import Dota2
from torch_frame.nn import TabTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='albert')
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--num_heads', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--encoder_pad_size', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = Dota2(root=path)
dataset.materialize()
dataset = dataset.shuffle()
# Split ratio following https://arxiv.org/abs/2012.06678
# 65% is used for training. 15% of is used for validation.
# The final reminder is used for testing.
train_dataset, val_dataset, test_dataset = dataset[:0.65], dataset[
    0.65:0.80], dataset[0.80:]

# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame.to(device)
val_tensor_frame = val_dataset.tensor_frame.to(device)
test_tensor_frame = test_dataset.tensor_frame.to(device)
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

# Set up model and optimizer
model = TabTransformer(
    channels=args.channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    encoder_pad_size=args.encoder_pad_size,
    attn_dropout=0.3,
    ffn_dropout=0.3,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        pred = model.forward(tf)
        loss = F.cross_entropy(pred, tf.y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    for tf in loader:
        pred = model(tf)
        pred_class = pred.argmax(dim=-1)

        all_labels.append(pred_class)
        all_preds.append(tf.y)
    all_labels = torch.cat(all_labels).cpu()
    all_preds = torch.cat(all_preds).cpu()

    # Compute the overall AUC
    overall_auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
    return overall_auc


metric = 'AUC'
best_val_metric = 0
best_test_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
          f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')

print(f'Best Val {metric}: {best_val_metric:.4f}, '
      f'Best Test {metric}: {best_test_metric:.4f}')
