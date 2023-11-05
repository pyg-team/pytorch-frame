import random
import pandas as pd
import os.path as osp

import torch
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.nn.models import FTTransformerLSTM
from torch_frame.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv("data/optiver/train.csv")
date_tf = dict()
num_stocks = 200
date_to_tf = {}
stock_target = {}
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'optiver')
for i in range(num_stocks):
    stock_target[f'stock_{i}'] = []
for date in data['date_id'].unique():
    df = data[data['date_id'] == date]
    #price = df[["stock_id", "target", "seconds_in_bucket"]]
    #price["seconds_in_bucket"] = price["seconds_in_bucket"]//10
    #new_index = range(0, 55)
    #for i in range(num_stocks):
    #    full_time_range_price = price[price["stock_id"] == i].set_index("seconds_in_bucket")
    #    stock_target[f"stock_{i}"].append(full_time_range_price.reindex(new_index)['target'].values)
    columns = [
        "date_id", "imbalance_buy_sell_flag",
        "seconds_in_bucket", "imbalance_size", "reference_price",
        "matched_size", "far_price", "near_price", "bid_price", "bid_size",
        "ask_price", "ask_size", "wap", "target"]
    covariates = pd.DataFrame(columns=columns) #df[columns]
    col_to_stype = {"imbalance_buy_sell_flag": stype.categorical, "seconds_in_bucket": stype.numerical,
                    "imbalance_size": stype.numerical, "reference_price": stype.numerical,
                    "matched_size": stype.numerical, "far_price": stype.numerical,
                    "near_price": stype.numerical, "bid_price": stype.numerical,
                    "bid_size": stype.numerical, "ask_price": stype.numerical,
                    "ask_size": stype.numerical, "wap": stype.numerical,
                    "target": stype.numerical}

    covariates_dataset = Dataset(covariates, col_to_stype, target_col="target")
    covariates_dataset.materialize(device=device, path=osp.join(path, f'day_{date}.pt'))
    date_to_tf[date] = covariates_dataset

#for i in range(num_stocks):
#    stock_target[f'stock_{i}'] = np.hstack(stock_target[f'stock_{i}'])
#price = pd.DataFrame(stock_target)

#price.to_csv(osp.join(path, "price.csv"))
price = pd.read_csv(osp.join(path, "price.csv"))
col_to_stype = {f"stock_{i}": stype.numerical for i in range(200)}
price_dataset = Dataset(price, col_to_stype)
price_dataset.materialize(device=device, path=osp.join(path, 'price.pt'))
print(price_dataset.df)
col_stats = torch.load(osp.join(path, 'col_stats.pt'))
print(col_stats)

dates = list(date_to_tf.keys())
random.shuffle(dates)
cutoff1 = int(0.8 * len(dates))
cutoff2 = int(0.9 * len(dates))
train_dates, val_dates, test_dates = dates[:cutoff1], dates[cutoff1:cutoff2], dates[cutoff2:] 
price = price_dataset.tensor_frame.to(device)
print("col names dict ", covariates_dataset.tensor_frame.col_names_dict)
model = FTTransformerLSTM(
    channels=args.channels,
    out_channels=1,
    num_layers=args.num_layers,
    lstm_channels=num_stocks,
    lstm_hidden=16,
    lstm_num_layers=2,
    lstm_out_channels=32,
    col_stats=col_stats,
    lstm_col_stats=price_dataset.col_stats,
    col_names_dict=covariates_dataset.tensor_frame.col_names_dict,
    lstm_col_names_dict=price_dataset.tensor_frame.col_names_dict,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


def train(dates) -> float:
    model.train()
    global price
    loss_accum = total_count = 0

    for d in dates:
        date_price = price[d*55: (d+1)*55]
        cov = date_to_tf[d].tensor_frame.to(device)
        loader = DataLoader(cov, batch_size=args.batch_size,
                          shuffle=True)
        for tf in tqdm(loader, desc=f'Date: {d}'):
            tf = tf.to(device)
            pred = model(tf, date_price)
            loss = loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(dates) -> float:
    model.eval()
    accum = total_count = 0

    for d in dates:
        date_price = price[d*55: (d+1)*55]
        cov = date_to_tf[d].tensor_frame.to(device)
        pred = model(cov, date_price)
        print("pred is ", pred)
        print("y is ", cov.y)
        import pdb
        pdb.set_trace()
        accum += float(
                F.mse_loss(pred.view(-1), cov.y.view(-1), reduction='sum'))
        total_count += len(cov.y)

    rmse = (accum / total_count)**0.5
    return rmse


metric = 'RMSE'
best_val_metric = float('inf')
best_test_metric = float('inf')

for epoch in range(1, args.epochs + 1):
    train_loss = train(train_dates)
    train_metric = test(train_dates)
    val_metric = test(val_dates)
    test_metric = test(test_dates)
    if val_metric < best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
        best_model_state = model.state_dict()
        torch.save(best_model_state, osp.join(path, 'best_model.pth'))
    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
          f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')
print(f'Best Val {metric}: {best_val_metric:.4f}, '
      f'Best Test {metric}: {best_test_metric:.4f}')

