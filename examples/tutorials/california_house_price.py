import argparse
import os.path as osp
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
# Please run `pip install transformers` to install the package
from transformers import AutoModel, AutoTokenizer

import torch_frame
from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.data.mapper import TextEmbeddingTensorMapper
from torch_frame.datasets import MultimodalTextBenchmark
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEncoder,
    MultiCategoricalEmbeddingEncoder,
)
from torch_frame.typing import TensorData, TextTokenizationOutputs

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="california_house_price")
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", type=str,
                    default="sentence-transformers/all-distilroberta-v1")
parser.add_argument("--pooling", type=str, default="mean",
                    choices=["mean", "cls"])
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()


class SimpleTextToEmbedding:
    def __init__(self, model: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device)
        self.device = device

    def __call__(self, sentences) -> Tensor:
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
        return out.last_hidden_state[:, 0, :].detach().cpu()


torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data",
                args.dataset)

# Load Dataset
dataset = MultimodalTextBenchmark(root=path, name=args.dataset)
print(dataset.df.columns)
text_encoder = SimpleTextToEmbedding(model=args.model, device=device)
mapper = TextEmbeddingTensorMapper(text_encoder, batch_size=50)
df = dataset.df
df['Summary'].fillna('', inplace=True)
df['Address'].fillna('', inplace=True)
df['Summary'] = mapper.forward(dataset.df['Summary'].values)
df['Address'] = mapper.forward(dataset.df['Address'].values)
print(df)

model_name = args.model.replace('/', '')
filename = f"{model_name}_data.pt"
dataset.materialize(path=osp.join(path, filename))

is_classification = dataset.task_type.is_classification

train_dataset, val_dataset, test_dataset = dataset.split()
if len(val_dataset) == 0:
    train_dataset, val_dataset = train_dataset[:0.9], train_dataset[0.9:]

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
    text_stype: text_stype_encoder,
    stype.multicategorical: MultiCategoricalEmbeddingEncoder(),
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
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
