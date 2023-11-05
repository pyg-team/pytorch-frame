import random
import pandas as pd
import os.path as osp

import torch
import torch.nn.functional as F
import tqdm
from torch_frame import stype
from torch_frame.data.dataset import Dataset
from torch_frame.data.loader import DataLoader
from torch_frame.nn.models import ResNetLSTM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv("data/optiver/train.csv")
date_tf = dict()
num_stocks = 200
date_to_tf = {}
for date in data['date_id'].unique():
    df = data[data['date_id'] == date]
    time = df[["stock_id", "target", "seconds_in_bucket"]]
    time["seconds_in_bucket"] = time["seconds_in_bucket"]//10
    new_index = range(0,55)
    stock_target = {}
    for i in range(num_stocks):
        stock_target[f"stock_{i}"] = time[time["stock_id"] == i].set_index("seconds_in_bucket")
        stock_target[f"stock_{i}"] = stock_target[f"stock_{i}"].reindex(new_index)['target'].values
    time = pd.DataFrame(stock_target)
    col_to_stype = {f"stock_{i}": stype.numerical for i in range(200)}
    time_dataset = Dataset(time, col_to_stype)
    time_dataset.materialize()
    columns = [
        "date_id", "imbalance_buy_sell_flag",
        "seconds_in_bucket", "imbalance_size", "reference_price",
        "matched_size", "far_price", "near_price", "bid_price", "bid_size",
        "ask_price", "ask_size", "wap", "target"]
    covariates = df[columns]
    col_to_stype = {"imbalance_buy_sell_flag": stype.categorical, "seconds_in_bucket": stype.numerical,
                    "imbalance_size": stype.numerical, "reference_price": stype.numerical,
                    "matched_size": stype.numerical, "far_price": stype.numerical,
                    "near_price": stype.numerical, "bid_price": stype.numerical,
                    "bid_size": stype.numerical, "ask_price": stype.numerical,
                    "ask_size": stype.numerical, "wap": stype.numerical,
                    "target": stype.numerical}

    covariates_dataset = Dataset(covariates, col_to_stype, target_col="target")
    covariates_dataset.materialize()
    date_to_tf[i] = (time_dataset, covariates_dataset)

col_to_stype = {"imbalance_buy_sell_flag": stype.categorical, "seconds_in_bucket": stype.numerical,
                    "imbalance_size": stype.numerical, "reference_price": stype.numerical,
                    "matched_size": stype.numerical, "far_price": stype.numerical,
                    "near_price": stype.numerical, "bid_price": stype.numerical,
                    "bid_size": stype.numerical, "ask_price": stype.numerical,
                    "ask_size": stype.numerical, "wap": stype.numerical,
                    "target": stype.numerical}
dataset = Dataset(data, col_to_stype)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'optiver')
dataset.materialize(path=osp.join(path, 'data.pt'))

torch.save(dataset.col_stats, osp.join(path, 'col_stats.pt'))