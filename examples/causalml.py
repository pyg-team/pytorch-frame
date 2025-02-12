import argparse
import copy
import os.path as osp

import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame import TensorFrame, stype
from torch_frame.data import DataLoader, Dataset
from torch_frame.datasets import IHDP
from torch_frame.nn.models import BCAUSS, CFR

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument('--model', type=str, default='cfr-mdd',
                    choices=["bcauss", "cfr-mdd"])
parser.add_argument("--feature-engineering", action="store_true", default=True)
parser.add_argument("--out-of-distribution", action="store_true",
                    default=False)
parser.add_argument("--lambda-reg", type=float, default=0.01,
                    help="l2 normalization score")
parser.add_argument("--split-num", type=int, default=0)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "ihdp")
dataset = IHDP(root=path, split_num=args.split_num)
ATE = dataset.get_att()
print(f"True Average Treatment Effect is {ATE}")

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset.materialize(path=osp.join(path, f"data_{args.split_num}.pt"))

dataset = dataset.shuffle()
within_sample_dataset, _, test_dataset = dataset.split()
# Train and Validation set is to compute within distribution metric
# Test set is to compute out of distribution metric
train_dataset, val_dataset = within_sample_dataset[:
                                                   0.7], within_sample_dataset[
                                                       0.7:]

# Counterfactual Set is to predict the outcome of the counterfactual
counterfactual_train_df = copy.deepcopy(train_dataset.df)
counterfactual_train_df['treated'] = 1 - counterfactual_train_df['treated']
counterfactual_train_df['target'] = counterfactual_train_df[
    'counterfactual_target']
counterfactual_train_dataset = Dataset(counterfactual_train_df,
                                       dataset.col_to_stype,
                                       target_col='target')

counterfactual_train_dataset.materialize(
    path=osp.join(path, f"counterfactual_train_data_{args.split_num}.pt"))

counterfactual_within_sample_df = copy.deepcopy(within_sample_dataset.df)
counterfactual_within_sample_df[
    'treated'] = 1 - counterfactual_within_sample_df['treated']
counterfactual_within_sample_df['target'] = counterfactual_within_sample_df[
    'counterfactual_target']
counterfactual_within_sample_dataset = Dataset(counterfactual_within_sample_df,
                                               dataset.col_to_stype,
                                               target_col='target')

counterfactual_within_sample_dataset.materialize(path=osp.join(
    path, f"counterfactual_within_sample_data_{args.split_num}.pt"))

counterfactual_val_df = copy.deepcopy(val_dataset.df)
counterfactual_val_df['treated'] = 1 - counterfactual_val_df['treated']
counterfactual_val_df['target'] = counterfactual_val_df[
    'counterfactual_target']
counterfactual_val_dataset = Dataset(counterfactual_val_df,
                                     dataset.col_to_stype, target_col='target')

counterfactual_val_dataset.materialize(
    path=osp.join(path, f"counterfactual_val_data_{args.split_num}.pt"))

counterfactual_test_df = copy.deepcopy(test_dataset.df)
counterfactual_test_df['treated'] = 1 - counterfactual_test_df['treated']
counterfactual_test_df['target'] = counterfactual_test_df[
    'counterfactual_target']
counterfactual_test_dataset = Dataset(counterfactual_test_df,
                                      dataset.col_to_stype,
                                      target_col='target')

counterfactual_test_dataset.materialize(
    path=osp.join(path, f"counterfactual_test_data_{args.split_num}.pt"))

train_tensor_frame = train_dataset.tensor_frame

treatment_idx = train_tensor_frame.col_names_dict[stype.categorical].index(
    'treated')
assert torch.all(
    torch.tensor(train_dataset.df['treated'].values, dtype=torch.long) ==
    train_tensor_frame.feat_dict[stype.categorical][:, treatment_idx])

counterfactual_train_tensor_frame = counterfactual_train_dataset.tensor_frame
counterfactual_train_tensor_frame.feat_dict[
    stype.categorical][:, treatment_idx] = torch.tensor(
        counterfactual_train_df['treated'].values)
val_tensor_frame = val_dataset.tensor_frame
counterfactual_val_tensor_frame = counterfactual_val_dataset.tensor_frame
counterfactual_val_tensor_frame.feat_dict[
    stype.categorical][:, treatment_idx] = torch.tensor(
        counterfactual_val_df['treated'].values)

within_sample_tensor_frame = within_sample_dataset.tensor_frame
counterfactual_within_sample_tensor_frame = (
    counterfactual_within_sample_dataset.tensor_frame)
counterfactual_within_sample_tensor_frame.feat_dict[
    stype.categorical][:, treatment_idx] = torch.tensor(
        counterfactual_within_sample_df['treated'].values)

test_tensor_frame = test_dataset.tensor_frame
counterfactual_test_tensor_frame = counterfactual_test_dataset.tensor_frame
counterfactual_test_tensor_frame.feat_dict[
    stype.categorical][:, treatment_idx] = torch.tensor(
        counterfactual_test_df['treated'].values)

train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
# val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
# test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

if args.model == 'cfr-mmd':
    model = CFR(
        channels=train_tensor_frame.num_cols - 1,
        hidden_channels=200,
        decoder_hidden_channels=100,
        out_channels=1,
        col_stats=dataset.col_stats if not args.feature_engineering else None,
        col_names_dict=train_tensor_frame.col_names_dict
        if not args.feature_engineering else None,
    ).to(device)
else:
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
lr_scheduler = ExponentialLR(optimizer, gamma=0.8)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        if args.model == 'cfr-mdd':
            out, ipm = model(tf, treatment_index=treatment_idx)
            treatment_val = tf.feat_dict[stype.categorical][:, treatment_idx]
            avg_treatment = torch.sum(treatment_val) / len(treatment_val)
            w_val = treatment_val / (2 * avg_treatment) + (
                1 - treatment_val) / (2 - 2 * avg_treatment)
            rmse = torch.sqrt(torch.mean(torch.square(tf.y - out.squeeze(-1))))
            loss = torch.mean(w_val * rmse) + ipm
        else:
            out, balance_score = model(tf, treatment_index=treatment_idx)
            treated_mask = tf.feat_dict[stype.categorical][:,
                                                           treatment_idx] == 1
            loss = (
                (torch.sum(treated_mask * torch.square(tf.y - out.squeeze(-1)))
                 + torch.sum(
                     ~treated_mask * torch.square(tf.y - out.squeeze(-1)))) /
                len(treated_mask) + balance_score)
        optimizer.zero_grad()
        loss.backward()
        if args.model == 'cfr-mdd':
            for name, param in model.named_parameters():
                if name.startswith('treatment_decoder') or name.startswith(
                        'control_decoder'):
                    loss += args.lambda_reg * torch.sum(param**2)
        loss_accum += float(loss) * len(out)
        total_count += len(out)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def eval(factual: TensorFrame, counterfactual: TensorFrame):
    model.eval()

    factual = factual.to(device)
    factual_effect, _ = model(factual, treatment_idx)
    # RMSE for factual predictions
    rmse_fact = torch.sqrt(torch.mean(torch.square(factual.y -
                                                   factual_effect)))

    counterfactual = counterfactual.to(device)
    counterfactual_effect, _ = model(counterfactual, treatment_idx)
    # RMSE for counterfactual predictions
    rmse_cfact = torch.sqrt(
        torch.mean(torch.square(counterfactual.y - counterfactual_effect)))
    eff_pred = counterfactual_effect - factual_effect
    t = factual.feat_dict[stype.categorical][:, treatment_idx]
    eff_pred[t > 0] = -eff_pred[t > 0]  # f(x, 1) - f(x, 0)
    ate_pred = torch.mean(eff_pred.squeeze(-1))
    bias_ate = torch.abs(ate_pred - ATE)

    pehe = torch.sqrt(torch.mean(torch.square(eff_pred - ATE)))
    return rmse_fact, rmse_cfact, bias_ate, pehe


best_val_error = float('inf')
best_test_error = float('inf')
best_val_pehe = float('inf')
best_test_pehe = float('inf')

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_rmse, train_rmse_cfact, train_error, train_pehe = eval(
        train_tensor_frame, counterfactual_train_tensor_frame)
    val_rmse, val_rmse_cfact, val_error, val_pehe = eval(
        val_tensor_frame, counterfactual_val_tensor_frame)
    test_rmse, test_rmse_cfact, test_error, test_pehe = eval(
        test_tensor_frame, counterfactual_test_tensor_frame)
    within_rmse, within_rmse_cfact, within_error, within_pehe = eval(
        within_sample_tensor_frame, counterfactual_within_sample_tensor_frame)

    if within_pehe < best_val_pehe:
        best_val_error = within_error
        best_test_error = test_error
        best_val_pehe = within_pehe
        best_test_pehe = test_pehe
    print(
        f'Train Loss: {train_loss:.4f}  Train Factual RMSE: {train_rmse:.4f} '
        f'Train Counterfactual RMSE: {train_rmse_cfact:.4f}, \n'
        f'Val Factual RMSE: {val_rmse:.4f} '
        f'Val Counterfactual RMSE: {val_rmse_cfact:.4f}, '
        f'Within Sample PEHE: {within_pehe:.4f}, '
        f'Within Sample Error: {within_error:.4f}, \n'
        f'Out of Distribution Factual RMSE: {val_rmse:.4f} '
        f'Out of Distribution Counterfactual RMSE: {val_rmse_cfact:.4f}, '
        f'Out of Distribution PEHE: {test_pehe:.4f}, '
        f'Out of Distribution Error: {test_error:.4f}, \n')

print(f'Best Within Sample Error: {best_val_error:.4f}, '
      f'Best Within Sample PEHE: {best_val_pehe}, \n'
      f'Best Out of Distribution Error: {best_test_error:.4f}, '
      f'Best Out of Distribution PEHE: {best_test_pehe:.4f}')
