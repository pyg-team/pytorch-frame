"""
Reported (reproduced) results of FT-Transformer
https://arxiv.org/abs/2106.11959

adult 86.0 (86.0)
helena 39.8 (39.2)
jannis 73.2 (71.6)

--------
Reported (reproduced) results of ResNet
https://arxiv.org/abs/2106.11959

 adult 86.0 ()
helena 39.8 ()
jannis 73.2 ()
"""
import argparse
import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_frame import stype
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
parser.add_argument(
    '--numerical_encoder_type', type=str, default='linear',
    choices=['linear', 'linearbucket', 'linearperiodic'],
    help='''The numerical encoder type to use: "linear",
    "linearbucket" or "linearperiodic".''')
parser.add_argument('--model_type', type=str, default='fttransformer',
                    choices=['fttransformer', 'resnet'],
                    help='The model type to use: "fttransformer" or "resnet".')
parser.add_argument('--channels', type=int, default=192)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=200)
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

encoder_config = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: numerical_encoder,
}

if args.model_type == 'fttransformer':
    model = FTTransformer(
        channels=args.channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
    ).to(device)
elif args.model_type == 'resnet':
    model = ResNet(
        channels=args.channels,
        out_channels=dataset.num_classes,
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
