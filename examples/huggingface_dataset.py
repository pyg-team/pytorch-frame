# This is an example to apply PyTorch Frame tabular learning
# with text embedders on the Hugging Face Spotify Tracks Dataset directly.
# You can specify different target columns such as "track_genre"
# (multiclass classification) or "loudness" (regression) etc.
# More dataset information please refer to:
# https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
# Please run `pip install transformers` to install the package
from transformers import AutoModel, AutoTokenizer

import torch_frame
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import DataLoader
from torch_frame.datasets import HuggingFaceDatasetDict
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
)

# Text Embedded with all-distilroberta-v1
# ================ track_genre ================
# Best Val Acc:  0.5169, Best Test Acc:  0.5130
# ================= loudness ==================
# Best Val RMSE: 2.1866, Best Test RMSE: 2.1577

class TextToEmbedding:
    def __init__(self, model: str, pooling: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device)
        self.device = device
        self.pooling = pooling

    def __call__(self, sentences: List[str]) -> Tensor:
        inputs = self.tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        for key in inputs:
            if isinstance(inputs[key], Tensor):
                inputs[key] = inputs[key].to(self.device)
        out = self.model(**inputs)
        mask = inputs["attention_mask"]
        if self.pooling == "mean":
            return (mean_pooling(out.last_hidden_state.detach(),
                                 mask).squeeze(1).cpu())
        elif self.pooling == "cls":
            return out.last_hidden_state[:, 0, :].detach().cpu()
        else:
            raise ValueError(f"{self.pooling} is not supported.")


def mean_pooling(last_hidden_state: Tensor, attention_mask) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)


parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--target_col", type=str, default="track_genre")
parser.add_argument(
    "--model",
    type=str,
    default="sentence-transformers/all-distilroberta-v1",
    choices=[
        "distilbert-base-uncased",
        "sentence-transformers/all-distilroberta-v1",
    ],
)
parser.add_argument("--pooling", type=str, default="mean",
                    choices=["mean", "cls"])
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "maharshipandya/spotify-tracks-dataset"

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data",
                dataset.replace("/", "_"))
os.makedirs(path, exist_ok=True)

col_to_stype = {
    "artists": torch_frame.categorical,
    "album_name": torch_frame.text_embedded,
    "track_name": torch_frame.text_embedded,
    "popularity": torch_frame.numerical,
    "duration_ms": torch_frame.numerical,
    "explicit": torch_frame.categorical,
    "danceability": torch_frame.numerical,
    "energy": torch_frame.numerical,
    "key": torch_frame.categorical,
    "loudness": torch_frame.numerical,
    "mode": torch_frame.categorical,
    "speechiness": torch_frame.numerical,
    "acousticness": torch_frame.numerical,
    "instrumentalness": torch_frame.numerical,
    "liveness": torch_frame.numerical,
    "valence": torch_frame.numerical,
    "tempo": torch_frame.numerical,
    "time_signature": torch_frame.categorical,
    "track_genre": torch_frame.categorical,
}
text_embedder = TextToEmbedding(model=args.model, pooling=args.pooling,
                                device=device)
col_to_text_embedder_cfg = TextEmbedderConfig(text_embedder=text_embedder,
                                              batch_size=10)
dataset = HuggingFaceDatasetDict(
    path=dataset,
    col_to_stype=col_to_stype,
    target_col=args.target_col,
    col_to_text_embedder_cfg=col_to_text_embedder_cfg,
)
model_name = args.model.replace('/', '')
filename = f"{model_name}_{args.target_col}_data.pt"
dataset.materialize(path=osp.join(path, filename))

dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset[:0.8], dataset[
    0.8:0.9], dataset[0.9:]

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
    stype.embedding: LinearEmbeddingEncoder(),
}

is_classification = dataset.task_type.is_classification

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
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


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

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)

    if is_classification and val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
    elif not is_classification and val_metric < best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
          f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}")

print(f"Best Val {metric}: {best_val_metric:.4f}, "
      f"Best Test {metric}: {best_test_metric:.4f}")
