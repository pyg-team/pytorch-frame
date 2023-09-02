"""
Reported (reproduced) results of FT-Transformer
https://arxiv.org/pdf/2106.11959.pdf

adult 86.0 (86.0)
helena 39.8 (39.2)
jannis 73.2 (71.6)
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
from torch_frame.datasets import Yandex
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEncoder,
    StypeWiseFeatureEncoder,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--channels', type=int, default=192)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = Yandex(root=path, name=args.dataset)
dataset.materialize()
train_indexes = dataset.df.index[dataset.df['split'] == 'train'].tolist()
train_dataset = dataset[train_indexes]
val_indexes = dataset.df.index[dataset.df['split'] == 'val'].tolist()
val_dataset = dataset[train_indexes]
test_indexes = dataset.df.index[dataset.df['split'] == 'test'].tolist()
test_dataset = dataset[test_indexes]

# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame.to(device)
val_tensor_frame = val_dataset.tensor_frame.to(device)
test_tensor_frame = test_dataset.tensor_frame.to(device)
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=1024)
test_loader = DataLoader(test_tensor_frame, batch_size=1024)


class FTTranformerModel(Module):
    def __init__(self):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=args.channels,
            col_stats=train_dataset.col_stats,
            col_names_dict=train_dataset.tensor_frame.col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            },
        )
        self.model = FTTransformer(
            in_channels=args.channels,
            out_channels=dataset.num_classes,
            num_cols=dataset.tensor_frame.num_cols,
            num_layers=args.num_layers,
        )

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        return self.model(x)


model = FTTranformerModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()
    loss_accum = 0

    for step, tf in enumerate(tqdm(train_loader)):
        pred = model(tf)
        loss = F.cross_entropy(pred, tf.y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss)
        optimizer.step()
    return loss_accum / (step + 1)


@torch.no_grad()
def eval(loader: DataLoader) -> float:
    model.eval()
    is_corret = []

    for tf in loader:
        pred = model(tf)
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
