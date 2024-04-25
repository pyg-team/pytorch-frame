import argparse
import copy
import os.path as osp

import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame import TensorFrame, stype
from torch_frame.data import DataLoader, Dataset
from torch_frame.datasets import IHDP
from torch_frame.nn.models import CFR

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--feature-engineering", action="store_true", default=True)
parser.add_argument("--out-of-distribution", action="store_true", default=True)
parser.add_argument("--lambda-reg", type=float, default=0.01,
                    help="l2 normalization score")
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "ihdp")
dataset = IHDP(root=path, split_num=0)
print(dataset.get_att())
ATT = dataset.get_att()
print(f"ATT is {ATT}")

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset.materialize(path=osp.join(path, "data.pt"))

dataset = dataset.shuffle()
if args.out_of_distribution:
    if dataset.split_col is None:
        train_dataset, val_dataset, test_dataset = dataset[:0.62], dataset[
            0.62:0.80], dataset[0.80:]
    else:
        train_dataset, _, test_dataset = dataset.split()
        train_dataset, val_dataset = train_dataset[:0.775], dataset[0.775:]
    # Calculating the validation dataset
    treated_df = val_dataset.df[(val_dataset.df['treated'] == 1)]
    treated_val_dataset = Dataset(treated_df, dataset.col_to_stype,
                                  target_col='target')
    control_df = copy.deepcopy(treated_df)
    control_df['treated'] = 0
    control_val_dataset = Dataset(control_df, dataset.col_to_stype,
                                  target_col='target')

    treated_val_dataset.materialize(path=osp.join(path, "treated_val_data.pt"))
    control_val_dataset.materialize(path=osp.join(path, "control_val_data.pt"))
    # Calculating the evaluation dataset
    treated_df = test_dataset.df[(test_dataset.df['treated'] == 1)]
    treated_test_dataset = Dataset(treated_df, dataset.col_to_stype,
                                   target_col='target')
    control_df = copy.deepcopy(treated_df)
    control_df['treated'] = 0
    control_test_dataset = Dataset(control_df, dataset.col_to_stype,
                                   target_col='target')

    treated_test_dataset.materialize(
        path=osp.join(path, "treated_test_data.pt"))
    control_test_dataset.materialize(
        path=osp.join(path, "control_test_data.pt"))
else:
    train_dataset = dataset

    # Calculating the evaluation dataset
    treated_df = dataset.df[(dataset.df['treated'] == 1)]
    treated_test_dataset = Dataset(treated_df, dataset.col_to_stype,
                                   target_col='target')
    control_df = copy.deepcopy(treated_df)
    control_df['treated'] = 0
    control_test_dataset = Dataset(control_df, dataset.col_to_stype,
                                   target_col='target')

    treated_test_dataset.materialize(
        path=osp.join(path, "treated_eval_data.pt"))
    control_test_dataset.materialize(
        path=osp.join(path, "control_eval_data.pt"))

train_tensor_frame = train_dataset.tensor_frame
treatment_idx = train_tensor_frame.col_names_dict[stype.categorical].index(
    'treated')
if args.out_of_distribution:
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    treated_val_tensor_frame = treated_val_dataset.tensor_frame
    # This is a bad hack. Currently the materialization logic would override
    # 1's to 0's due to 1's being the popular class
    treated_val_tensor_frame.feat_dict[stype.categorical][:,
                                                          treatment_idx] = 1.
    control_val_tensor_frame = control_val_dataset.tensor_frame

treated_test_tensor_frame = treated_test_dataset.tensor_frame
# This is a bad hack. Currently the materialization logic would override 1's
# to 0's due to 1's being the popular class
treated_test_tensor_frame.feat_dict[stype.categorical][:, treatment_idx] = 1.
control_test_tensor_frame = control_test_dataset.tensor_frame

train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
# val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
# test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

model = CFR(
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
        out, ipm = model(tf, treatment_index=treatment_idx)
        treatment_val = tf.feat_dict[stype.categorical][:, treatment_idx]
        avg_treatment = torch.sum(treatment_val) / len(treatment_val)
        w_val = treatment_val / (2 * avg_treatment) + (1 - treatment_val) / (
            2 - 2 * avg_treatment)
        loss = torch.mean(w_val * (tf.y - out.squeeze(-1))) + ipm
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if name.startswith('treatment_decoder') or name.startswith(
                    'control_decoder'):
                loss += args.lambda_reg * torch.sum(param**2)
        loss_accum += float(loss) * len(out)
        total_count += len(out)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def eval(treated: TensorFrame, control: TensorFrame) -> float:
    model.eval()

    treated = treated.to(device)
    treated_effect, _ = model(treated, treatment_idx)

    control = control.to(device)
    control_effect, _ = model(control, treatment_idx)

    return torch.abs(ATT - torch.mean(treated_effect - control_effect))


best_val_metric = float('inf')
best_test_metric = float('inf')

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    error = eval(treated_test_tensor_frame, control_test_tensor_frame)
    if args.out_of_distribution:
        val_error = eval(treated_val_tensor_frame, control_val_tensor_frame)
        if val_error < best_val_metric:
            best_val_metric = val_error
            best_test_metric = error
        print(
            f'Train Loss: {train_loss:.4f}  Val Error_ATT: {val_error:.4f},\n'
            f' Error_ATT: {error:.4f}\n')
    else:
        print(f'Train Loss: {train_loss:.4f}  Error_ATT: {error:.4f},\n')

if args.out_of_distribution:
    print(f'Best Val Error: {best_val_metric:.4f}, '
          f'Best Test Error: {best_test_metric:.4f}')
