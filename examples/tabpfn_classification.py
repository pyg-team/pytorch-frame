import argparse
import os.path as osp

import numpy as np
import torch
from tabpfn import TabPFNClassifier
# Please run `pip install tabpfn` to install the package
from tqdm import tqdm

from torch_frame.data import DataLoader
from torch_frame.datasets import (
    ForestCoverType,
    KDDCensusIncome,
    Mushroom,
    Titanic,
)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset', type=str, default="Titanic",
    choices=["Titanic", "Mushroom", "ForestCoverType", "KDDCensusIncome"])
parser.add_argument('--train_batch_size', type=int, default=4096)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)

if args.dataset == "Titanic":
    dataset = Titanic(root=path)
elif args.dataset == "ForestCoverType":
    dataset = ForestCoverType(root=path)
elif args.dataset == "KDDCensusIncome":
    dataset = KDDCensusIncome(root=path)
else:
    dataset = Mushroom(root=path)

dataset.materialize()
assert dataset.task_type.is_classification
dataset = dataset.shuffle()
train_dataset, test_dataset = dataset[:0.9], dataset[0.9:]
train_tensor_frame = train_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(
    train_tensor_frame,
    batch_size=args.train_batch_size,
    shuffle=True,
)
X_train = []
train_data = next(iter(train_loader))
for stype in train_data.stypes:
    X_train.append(train_data.feat_dict[stype])
X_train: torch.Tensor = torch.cat(X_train, dim=1)
clf = TabPFNClassifier()
clf.fit(X_train, train_data.y)
test_loader = DataLoader(test_tensor_frame, batch_size=args.test_batch_size)


@torch.no_grad()
def test() -> float:
    accum = total_count = 0
    for test_data in tqdm(test_loader):
        X_test = []
        for stype in train_data.stypes:
            X_test.append(test_data.feat_dict[stype])
        X_test = torch.cat(X_test, dim=1)
        pred: np.ndarray = clf.predict_proba(X_test)
        pred_class = pred.argmax(axis=-1)
        accum += float((test_data.y.numpy() == pred_class).sum())
        total_count += len(test_data.y)

    return accum / total_count


acc = test()
print(f"Accuracy: {acc:.4f}")
