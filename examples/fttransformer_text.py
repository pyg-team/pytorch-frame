import argparse
import os.path as osp
from typing import Any, List

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm
import torch_frame

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data import DataLoader
from torch_frame.datasets import MultimodalTextBenchmark
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
    LinearModelEncoder,
)
from transformers import AutoTokenizer, DistilBertModel

# Text embedded:
# ============== wine_reviews ===============
# Best Val Acc: 0.7946, Best Test Acc: 0.7878


class PretrainedTextEncoder:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = SentenceTransformer('all-distilroberta-v1')

    def __call__(self, sentences: List[str]) -> Tensor:
        embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                       convert_to_tensor=True)
        return embeddings.to(self.device)


class Tokenizer:
    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def __call__(self, sentences: List[str]) -> torch.nested.nested_tensor:
        res = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence)
            res.append(inputs['input_ids'])
        return torch.nested.nested_tensor(res)


class DistilBert:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    
    def __call__(self, input_ids: Tensor) -> Tensor:
        breakpoint()
        return self.model(input_ids=input_ids)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wine_reviews')
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_tokenizer', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)


if not args.use_tokenizer:
    text_encoder = PretrainedTextEncoder(device=device)
    text_embedder_cfg=TextEmbedderConfig(text_embedder=text_encoder,
                                         batch_size=5),
    kwargs = dict(text_embedder_cfg=text_embedder_cfg, text_stype=torch_frame.text_embedded)
else:
    text_tokenizer = Tokenizer(device=device)
    text_tokenizer_cfg = TextTokenizerConfig(text_tokenizer=text_tokenizer)
    kwargs = dict(text_tokenizer_cfg=text_tokenizer_cfg, text_stype=torch_frame.text_tokenized)

dataset = MultimodalTextBenchmark(
    root=path,
    name=args.dataset,
    **kwargs,
)

dataset.materialize(path=osp.join(path, 'data.pt'))

is_classification = dataset.task_type.is_classification

train_dataset = dataset.get_split_dataset('train')[:0.9]
val_dataset = dataset.get_split_dataset('train')[0.9:]
test_dataset = dataset.get_split_dataset('test')

# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame.to(device)
val_tensor_frame = val_dataset.tensor_frame.to(device)
test_tensor_frame = test_dataset.tensor_frame.to(device)
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

if not args.use_tokenizer:
    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.text_embedded: LinearEmbeddingEncoder(in_channels=768)
    }
else:
    distil_model = DistilBert(device=device)
    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.text_tokenized: LinearModelEncoder(model=distil_model, in_channels=768)
    }

model = FTTransformer(
    channels=args.channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
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
        pred = model(tf)
        if is_classification:
            pred_class = pred.argmax(dim=-1)
            accum += float((tf.y == pred_class).sum())
        else:
            accum += float(
                F.mse_loss(pred.view(-1), tf.y.view(-1), reduction='sum'))
        total_count += len(tf.y)

    if is_classification:
        accuracy = accum / total_count
        return accuracy
    else:
        rmse = (accum / total_count)**0.5
        return rmse


if is_classification:
    metric = 'Acc'
    best_val_metric = 0
    best_test_metric = 0
else:
    metric = 'RMSE'
    best_val_metric = float('inf')
    best_test_metric = float('inf')

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

    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
          f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')

print(f'Best Val {metric}: {best_val_metric:.4f}, '
      f'Best Test {metric}: {best_test_metric:.4f}')
