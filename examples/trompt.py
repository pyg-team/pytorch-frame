"""Reported (reproduced) results of of Trompt model based on Tables 9--20 of
the original paper: https://arxiv.org/abs/2305.18446.

electricity (A4): 84.50 (84.17)
eye_movements (A5): 64.25 (63.02)
MagicTelescope (B2): 86.30 (86.93)
bank-marketing (B4): 79.36 (80.59)
california (B5): 89.09 (89.17)
credit (B7): 75.84 (76.01)
pol (B14): 98.49 (98.82)
jannis (mathcal B4): 79.54 (80.29)

Reported results of Trompt model on Yandex dataset
helena : 37.90
jannis : 72.98
"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame.data import DataLoader
from torch_frame.datasets import TabularBenchmark
from torch_frame.nn import Trompt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="california")
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--num_prompts", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data",
                args.dataset)
dataset = TabularBenchmark(root=path, name=args.dataset)
dataset.materialize()
# Only support classification training/eval for now.
# TODO: support regression tasks.
assert dataset.task_type.is_classification
dataset = dataset.shuffle()
# Split ratio following https://arxiv.org/abs/2207.08815
# 70% is used for training. 30% of the remaining is used for validation.
# The final reminder is used for testing.
train_dataset, val_dataset, test_dataset = (
    dataset[:0.7],
    dataset[0.7:0.79],
    dataset[0.79:],
)

# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

# Set up model and optimizer
model = Trompt(
    channels=args.channels,
    out_channels=dataset.num_classes,
    num_prompts=args.num_prompts,
    num_layers=args.num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
).to(device)
model = torch.compile(model, dynamic=True) if args.compile else model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        tf = tf.to(device)
        # [batch_size, num_layers, num_classes]
        out = model.forward_stacked(tf)
        num_layers = out.size(1)
        # [batch_size * num_layers, num_classes]
        pred = out.view(-1, dataset.num_classes)
        y = tf.y.repeat_interleave(num_layers)
        # Layer-wise logit loss
        loss = F.cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    accum = total_count = 0

    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        pred_class = pred.argmax(dim=-1)
        accum += float((tf.y == pred_class).sum())
        total_count += len(tf.y)

    return accum / total_count


best_val_acc = 0
best_test_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
    lr_scheduler.step()

print(f"Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}")
