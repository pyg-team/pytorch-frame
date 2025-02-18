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

# Use TF32 for faster matrix multiplication on Ampere GPUs.
# https://dev-discuss.pytorch.org/t/pytorch-and-tensorfloat32/504
torch.set_float32_matmul_precision('high')

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
dataset.materialize(path=osp.join(path, 'materialized_data.pt'))
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
train_loader = DataLoader(
    train_tensor_frame,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_tensor_frame,
    batch_size=args.batch_size,
    pin_memory=True,
)
test_loader = DataLoader(
    test_tensor_frame,
    batch_size=args.batch_size,
    pin_memory=True,
)

model = Trompt(
    channels=args.channels,
    out_channels=dataset.num_classes,
    num_prompts=args.num_prompts,
    num_layers=args.num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
).to(device)
model = torch.compile(model, dynamic=True) if args.compile else model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)


def train(epoch: int) -> torch.Tensor:
    model.train()
    loss_accum = torch.zeros(1, device=device, dtype=torch.float32).squeeze_()
    total_count = 0

    for tf in tqdm(train_loader, desc=f"Epoch {epoch:3d}"):
        tf = tf.to(device, non_blocking=True)
        # [batch_size, num_layers, num_classes]
        out = model(tf)
        batch_size, num_layers, num_classes = out.size()
        # [batch_size * num_layers, num_classes]
        pred = out.view(-1, num_classes)
        y = tf.y.repeat_interleave(
            num_layers,
            output_size=num_layers * batch_size,
        )
        # Layer-wise logit loss
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_count += len(tf.y)
        loss *= len(tf.y)
        loss_accum += loss

    lr_scheduler.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader, desc: str) -> torch.Tensor:
    model.eval()
    accum = torch.zeros(1, device=device, dtype=torch.long).squeeze_()
    total_count = 0

    for tf in tqdm(loader, desc=desc):
        tf = tf.to(device, non_blocking=True)
        pred = model(tf).mean(dim=1)
        pred_class = pred.argmax(dim=-1)
        accum += (tf.y == pred_class).sum()
        total_count += len(tf.y)

    return accum / total_count


best_val_acc = 0.0
best_test_acc = 0.0
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_acc = test(train_loader, "Eval (train)")
    val_acc = test(val_loader, "Eval (val)")
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test(test_loader, "Eval (test)")

    print(f"Train Loss: {train_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, "
          f"Val Acc: {val_acc:.4f}, "
          f"Test Acc: {best_test_acc:.4f}")

print(f"Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}")
