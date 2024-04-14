import argparse
import os.path as osp

import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.datasets import Jobs
from torch_frame.nn.models import BCAUSS

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "jobs")
dataset = Jobs(root=path)

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset.materialize(path=osp.join(path, "data.pt"))

dataset = dataset.shuffle()
tensor_frame = dataset.tensor_frame
train_loader = DataLoader(tensor_frame, batch_size=args.batch_size,
                          shuffle=True)

model = BCAUSS(
    channels=tensor_frame.num_cols - 1,
    hidden_channels=200,
    decoder_hidden_channels=100,
    out_channels=1,
    col_stats=dataset.col_stats,
    col_names_dict=tensor_frame.col_names_dict,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

is_classification = True


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        treatment_idx = tf.col_names_dict[stype.categorical].index('treated')
        out, balance_score, treated_mask = model.forward(
            tf, treatment_index=treatment_idx)
        loss = torch.mean(treated_mask *
                          torch.square(tf.y - out.squeeze(-1))) + balance_score
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(out)
        total_count += len(out)
        optimizer.step()
    return loss_accum / total_count


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)

    print(f'Train Loss: {train_loss:.4f}\n')
