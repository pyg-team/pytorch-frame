import argparse
import os.path as osp

import numpy as np
import torch
from tabpfn import TabPFNClassifier
# Please run `pip install tabpfn` to install the package
from tqdm import tqdm

from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.datasets import (
    ForestCoverType,
    KDDCensusIncome,
    Mushroom,
    Titanic,
)
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearBucketEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    StypeWiseFeatureEncoder,
)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset', type=str, default="Titanic",
    choices=["Titanic", "Mushroom", "ForestCoverType", "KDDCensusIncome"])
parser.add_argument('--train_batch_size', type=int, default=4096)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out_channels', type=int, default=16)
parser.add_argument(
    '--numerical_encoder_type',
    type=str,
    default='linear',
    choices=['linear', 'linearbucket', 'linearperiodic'],
)
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

if args.numerical_encoder_type == 'linear':
    numerical_encoder = LinearEncoder()
elif args.numerical_encoder_type == 'linearbucket':
    numerical_encoder = LinearBucketEncoder()
elif args.numerical_encoder_type == 'linearperiodic':
    numerical_encoder = LinearPeriodicEncoder()
else:
    raise ValueError(
        f'Unsupported encoder type: {args.numerical_encoder_type}')

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: numerical_encoder,
}

encoder = StypeWiseFeatureEncoder(
    out_channels=args.out_channels,
    col_stats=dataset.col_stats,
    col_names_dict=dataset.tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
)
encoder.eval()

train_dataset, test_dataset = dataset[:0.9], dataset[0.9:]
train_tensor_frame = train_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(
    train_tensor_frame,
    batch_size=args.train_batch_size,
    shuffle=True,
)
train_data = next(iter(train_loader))
X_train, col_names = encoder(train_data)
X_train = X_train.reshape(X_train.size(0), -1).detach()
clf = TabPFNClassifier()
clf.fit(X_train, train_data.y)
test_loader = DataLoader(test_tensor_frame, batch_size=args.test_batch_size)


@torch.no_grad()
def test() -> float:
    accum = total_count = 0
    for test_data in tqdm(test_loader):
        X_test, _ = encoder(test_data)
        X_test = X_test.reshape(X_test.size(0), -1).detach()
        pred: np.ndarray = clf.predict_proba(X_test)
        pred_class = pred.argmax(axis=-1)
        accum += float((test_data.y.numpy() == pred_class).sum())
        total_count += len(test_data.y)

    return accum / total_count


acc = test()
print(f"Accuracy: {acc:.4f}")
