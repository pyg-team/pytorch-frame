"""
Expected default accuracy of Trompt based on Tables 9, 10, and 11 of the paper
https://arxiv.org/abs/2305.18446

comass (A2): 78.59
electricity (A4): 84.50
eye_movements (A5): 64.25
california (B5): 89.09
credit (B7): 75.84
jannis (B11): 76.89
pol (B14): 98.49
"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm

from torch_frame import stype
from torch_frame.data import DataLoader, TensorFrame
from torch_frame.datasets import TabularBenchmark
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
    Trompt,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='california')
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--num_prompts', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = TabularBenchmark(root=path, name=args.dataset)
dataset.materialize()
dataset = dataset.shuffle()
# Split ratio following https://arxiv.org/abs/2207.08815
# 70% is used for training. 30% of the remaining is used for validation.
# The final reminder is used for testing.
train_dataset, val_dataset, test_dataset = dataset[:0.7], dataset[
    0.7:0.79], dataset[0.79:]
# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame.to(device)
val_tensor_frame = val_dataset.tensor_frame.to(device)
test_tensor_frame = test_dataset.tensor_frame.to(device)
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=1024)
test_loader = DataLoader(test_tensor_frame, batch_size=1024)


class TromptModel(Module):
    def __init__(self):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=args.channels,
            col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(post_act='relu'),
            },
        )
        self.model = Trompt(
            in_channels=args.channels,
            out_channels=dataset.num_classes,
            num_cols=train_dataset.tensor_frame.num_cols,
            num_prompts=args.num_prompts,
            num_layers=args.num_layers,
        )

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        # [batch_size, num_layers, num_classes]
        return self.model(x)


trompt_model = TromptModel().to(device)
optimizer = torch.optim.Adam(trompt_model.parameters(), lr=args.lr)


def train() -> float:
    trompt_model.train()
    loss_accum = 0

    for step, tf in enumerate(tqdm(train_loader)):
        # [batch_size, num_layers, num_classes]
        out = trompt_model(tf)
        num_layers = out.size(1)
        # [batch_size * num_layers, num_classes]
        pred = out.view(-1, dataset.num_classes)
        y = tf.y.repeat_interleave(num_layers)
        # Layer-wise logit loss
        loss = F.cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss)
        optimizer.step()
    return loss_accum / (step + 1)


@torch.no_grad()
def eval(loader: DataLoader) -> float:
    trompt_model.eval()
    is_corret = []

    for tf in loader:
        # [batch_size, num_layers, num_classes]
        out = trompt_model(tf)
        # Mean pooling across layers
        # [batch_size, num_layers, num_classes] -> [batch_size, num_classes]
        pred = out.mean(dim=1)
        pred_class = pred.argmax(dim=-1)
        is_corret.append((tf.y == pred_class).detach().cpu())

    is_correct_cat = torch.cat(is_corret)
    return float(is_correct_cat.sum()) / len(is_correct_cat)


best_val_acc = 0
best_test_acc = 0
for epoch in range(args.epochs):
    print(f"=====epoch {epoch}")
    loss = train()
    print(f'Train loss: {loss}')
    train_acc = eval(train_loader)
    val_acc = eval(val_loader)
    test_acc = eval(test_loader)
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    print(f'Train acc: {train_acc}, val acc: {val_acc}, test acc: {test_acc}')

print(f'Best val acc: {best_val_acc}, best test acc: {best_test_acc}')
