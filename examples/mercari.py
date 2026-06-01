"""Example of a regression task using sentence-transformers to embed item name
and item descriptions of Mercari Price Suggestion Challenage dataset
https://www.kaggle.com/c/mercari-price-suggestion-challenge/.

Train Loss: 540.2466, Train RMSE: 22.6720, Val RMSE: 26.2494
Private Score: 0.50207 Public Score: 0.50156.
"""
from __future__ import annotations

import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import DataLoader
from torch_frame.datasets import Mercari
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
)
from torch_frame.nn.encoder.stype_encoder import \
    MultiCategoricalEmbeddingEncoder  # noqa


class PretrainedTextEncoder:
    def __init__(self, device: torch.device) -> None:
        self.model = SentenceTransformer("all-distilroberta-v1", device=device)

    def __call__(self, sentences: list[str]) -> Tensor:
        # Inference on GPU (if available)
        embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                       convert_to_tensor=True)
        # Map back to CPU
        return embeddings.cpu()


parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "mercari")
text_encoder = PretrainedTextEncoder(device=device)
dataset = Mercari(
    root=path,
    num_rows=5000,  # set num_rows to use a subset of the dataset
    col_to_text_embedder_cfg=TextEmbedderConfig(text_embedder=text_encoder,
                                                batch_size=5),
)

dataset.materialize(path=osp.join(path, "data.pt"))

is_classification = dataset.task_type.is_classification

# Use the pre-defined split when using the entire dataset
if dataset.num_rows == 4_943_260:
    train_dataset, val_dataset, test_dataset = dataset.split()
else:
    train_dataset = dataset[:0.8]
    val_dataset = dataset[0.8:0.9]
    test_dataset = dataset[0.9:]

# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.multicategorical: MultiCategoricalEmbeddingEncoder(),
    stype.embedding: LinearEmbeddingEncoder(),
}

if is_classification:
    output_channels = dataset.num_classes
else:
    output_channels = 1

model = FTTransformer(
    channels=args.channels,
    out_channels=output_channels,
    num_layers=args.num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
).to(device)
model = torch.compile(model, dynamic=True) if args.compile else model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        tf = tf.to(device)
        pred = model(tf)
        if is_classification:
            loss = F.cross_entropy(pred, tf.y)
        else:
            loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    accum = total_count = 0

    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if is_classification:
            pred_class = pred.argmax(dim=-1)
            accum += float((tf.y == pred_class).sum())
        else:
            accum += float(
                F.mse_loss(pred.view(-1), tf.y.view(-1), reduction="sum"))
        total_count += len(tf.y)

    if is_classification:
        accuracy = accum / total_count
        return accuracy
    else:
        rmse = (accum / total_count)**0.5
        return rmse


if is_classification:
    metric = "Acc"
    best_val_metric = 0
    best_test_metric = 0
else:
    metric = "RMSE"
    best_val_metric = float("inf")
    best_test_metric = float("inf")

best_model_state = None

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)

    if is_classification and val_metric > best_val_metric:
        best_val_metric = val_metric
        best_model_state = model.state_dict()
    elif not is_classification and val_metric < best_val_metric:
        best_val_metric = val_metric
        best_model_state = model.state_dict()
        torch.save(best_model_state, osp.join(path, "best_model.pth"))

    print(f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
          f"Val {metric}: {val_metric:.4f}")
    lr_scheduler.step()

print(f"Best Val {metric}: {best_val_metric:.4f}, ")

# Generate prediction csv
model.load_state_dict(best_model_state)
with torch.inference_mode():
    model.eval()
    accum = total_count = 0
    all_preds = []
    for tf in test_loader:
        tf = tf.to(device)
        pred = model(tf)
        all_preds.append(pred.cpu().numpy())
pred = np.concatenate(all_preds).flatten()
df = dataset.df[dataset.df["split_col"] == 2]
df = df[["test_id"]]
df["test_id"] = df["test_id"].astype(int)
df[dataset.target_col] = pred
df.to_csv("submission.csv")
