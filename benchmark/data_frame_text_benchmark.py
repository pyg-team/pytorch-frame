import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameTextBenchmark
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
    MultiCategoricalEmbeddingEncoder,
)
from torch_frame.nn.encoder.stype_encoder import TimestampEncoder
from torch_frame.typing import TaskType

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type',
    type=str,
    choices=[
        'binary_classification',
        'multiclass_classification',
        'regression',
    ],
    default='binary_classification',
)
parser.add_argument(
    '--scale',
    type=str,
    choices=['small', 'medium', 'large'],
    default='small',
)
parser.add_argument(
    '--model',
    type=str,
    default='sentence-transformers/all-distilroberta-v1',
    choices=[
        'distilbert-base-uncased',
        'sentence-transformers/all-distilroberta-v1',
    ],
)
parser.add_argument("--finetune", action="store_true")
parser.add_argument(
    '--idx',
    type=int,
    default=0,
    help='The index of the dataset within DataFrameBenchmark',
)
args = parser.parse_args()


class TextToEmbedding:
    def __init__(self, model: str, device: torch.device):
        self.model_name = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device)

    def __call__(self, sentences: list[str]) -> Tensor:
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
        # [batch_size, max_length or batch_max_length]
        # Value is either one or zero, where zero means that
        # the token is not attended to other tokens.
        mask = inputs["attention_mask"]
        return (mean_pooling(out.last_hidden_state.detach(),
                             mask).squeeze(1).cpu())


def mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')

if not args.finetune:
    text_encoder = TextToEmbedding(model=args.model, device=device)
    text_stype = torch_frame.text_embedded
    kwargs = {
        "text_stype":
        text_stype,
        "col_to_text_embedder_cfg":
        TextEmbedderConfig(text_embedder=text_encoder, batch_size=5),
    }
else:
    raise ValueError('Finetune not supported')

dataset = DataFrameTextBenchmark(
    root=path,
    task_type=TaskType(args.task_type),
    scale=args.scale,
    idx=args.idx,
    **kwargs,
)

model_name = args.model.replace('/', '')
filename = f"{model_name}_{text_stype.value}_data.pt"
dataset.materialize(path=osp.join(path, filename))
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=512, shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=512)
test_loader = DataLoader(test_tensor_frame, batch_size=512)

text_stype_encoder = LinearEmbeddingEncoder()

stype_encoder_dict = {
    torch_frame.categorical:
    EmbeddingEncoder(),
    torch_frame.numerical:
    LinearEncoder(),
    # If text_stype is text_embedded,
    # it becomes embedding after materialization
    text_stype.parent:
    text_stype_encoder,
    torch_frame.multicategorical:
    MultiCategoricalEmbeddingEncoder(),
    torch_frame.timestamp:
    TimestampEncoder()
}

is_classification = dataset.task_type.is_classification
if is_classification:
    output_channels = dataset.num_classes
else:
    output_channels = 1

model = FTTransformer(
    channels=128,
    out_channels=output_channels,
    num_layers=4,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


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
        breakpoint()
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

for epoch in range(1, 21):
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
