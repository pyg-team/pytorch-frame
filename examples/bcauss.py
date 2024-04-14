"""Reported (reproduced) E_ATT of BCAUSS based on Table 1 of the paper.
BAUSS + in_sample 0.02 (0.0284)
"""
import argparse
import copy
import os.path as osp

import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame import TensorFrame, stype
from torch_frame.data import DataLoader, Dataset
from torch_frame.datasets import Jobs
from torch_frame.nn.models import BCAUSS

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=3000)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--feature_engineering", action="store_true", default=True)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "jobs")
dataset = Jobs(root=path, feature_engineering=args.feature_engineering)
ATT = dataset.get_att()
print(f"ATT is {ATT}")

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset.materialize(path=osp.join(path, "data.pt"))

dataset = dataset.shuffle()
"""
if dataset.split_col is None:
    train_dataset, val_dataset, test_dataset = dataset[:0.62], dataset[
    0.62:0.80], dataset[0.80:]
else:
    train_dataset, val_dataset, test_dataset = dataset.split()
"""
train_dataset = dataset
treated_df = dataset.df[(dataset.df['source'] == 1)
                        & (dataset.df['treated'] == 1)]
treated_eval_dataset = Dataset(treated_df, dataset.col_to_stype,
                               target_col='target')
control_df = copy.deepcopy(treated_df)
control_df['treated'] = 0
control_eval_dataset = Dataset(control_df, dataset.col_to_stype,
                               target_col='target')

treated_eval_dataset.materialize(path=osp.join(path, "treated_eval_data.pt"))
control_eval_dataset.materialize(path=osp.join(path, "control_eval_data.pt"))

train_tensor_frame = train_dataset.tensor_frame
treatment_idx = train_tensor_frame.col_names_dict[stype.categorical].index(
    'treated')
# val_tensor_frame = val_dataset.tensor_frame
# test_tensor_frame = test_dataset.tensor_frame
treated_eval_tensor_frame = treated_eval_dataset.tensor_frame
# This is a bad hack. Currently the materialization logic would override 1's
# to 0's due to 1's being the popular class
treated_eval_tensor_frame.feat_dict[stype.categorical][:, treatment_idx] = 1.
control_eval_tensor_frame = control_eval_dataset.tensor_frame

train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
# val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
# test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

model = BCAUSS(
    channels=train_tensor_frame.num_cols - 1,
    hidden_channels=200,
    decoder_hidden_channels=100,
    out_channels=1,
    col_stats=dataset.col_stats if not args.feature_engineering else None,
    col_names_dict=train_tensor_frame.col_names_dict
    if not args.feature_engineering else None,
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

is_classification = True


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
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


@torch.no_grad()
def test(treated: TensorFrame, control: TensorFrame) -> float:
    model.eval()

    treated = treated.to(device)
    treated_effect, _, _ = model(treated, treatment_idx)

    control = control.to(device)
    control_effect, _, _ = model(control, treatment_idx)

    return torch.abs(ATT - torch.mean(treated_effect - control_effect))


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    error = test(treated_eval_tensor_frame, control_eval_tensor_frame)

    print(f'Train Loss: {train_loss:.4f}  Error_ATT: {error:.4f},\n')
