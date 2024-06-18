import argparse
import os
import os.path as osp
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import DataLoader
from torch_frame.datasets import MultimodalTextBenchmark
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
)

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--service",
    type=str,
    default="voyageai",
    choices=["openai", "cohere", "voyageai"],
)
parser.add_argument("--dataset", type=str, default="wine_reviews")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument('--compile', action='store_true')
args = parser.parse_args()

# Notice that there are 568,454 rows and 2 text columns, it will
# cost some money to get the text embeddings by using OpenAI API
if args.service == "openai":
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError("OpenAI API key is not specified.")
elif args.service == "cohere":
    api_key = args.api_key or os.environ.get("COHERE_API_KEY", None)
    if api_key is None:
        raise ValueError("Cohere API key is not specified.")
else:
    api_key = args.api_key or os.environ.get("VOYAGE_API_KEY", None)
    if api_key is None:
        raise ValueError("Voyageai API key is not specified.")


class OpenAIEmbedding:
    dimension: int = 1536
    text_embedder_batch_size: int = 25

    def __init__(self, model: str = "text-embedding-ada-002"):
        # Please run `pip install openai` to install the package
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, sentences: List[str]) -> Tensor:
        from openai import Embedding

        items: List[Embedding] = self.client.embeddings.create(
            input=sentences, model=self.model).data
        assert len(items) == len(sentences)
        embeddings = [
            torch.FloatTensor(item.embedding).view(1, -1) for item in items
        ]
        return torch.cat(embeddings, dim=0)


class CohereEmbedding:
    dimension: int = 1024
    text_embedder_batch_size: int = 1000

    def __init__(self, model: str = "embed-english-v3.0"):
        # Please run `pip install cohere` to install the package
        import cohere  # noqa

        self.model = model
        self.co = cohere.Client(api_key)

    def __call__(self, sentences: List[str]) -> Tensor:
        from cohere import EmbedResponse

        response: EmbedResponse = self.co.embed(model=self.model,
                                                texts=sentences,
                                                input_type="classification")
        assert len(response.embeddings) == len(sentences)
        embeddings = torch.tensor(response.embeddings)
        return embeddings


class VoyageaiEmbedding:
    dimension: int = 1024
    text_embedder_batch_size: int = 8

    def __init__(self, model: str = "voyage-01"):
        # Please run `pip install voyageai` to install the package
        self.model = model

    def __call__(self, sentences: List[str]) -> Tensor:
        import voyageai  # noqa

        voyageai.api_key = api_key
        from voyageai import get_embeddings

        items: List[List[float]] = get_embeddings(sentences, model=self.model)
        assert len(items) == len(sentences)
        embeddings = [torch.FloatTensor(item).view(1, -1) for item in items]
        return torch.cat(embeddings, dim=0)


torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(
    osp.dirname(osp.realpath(__file__)),
    "..",
    "data",
    "amazon_fine_food_reviews",
)
os.makedirs(path, exist_ok=True)
if args.service == "openai":
    text_encoder = OpenAIEmbedding()
elif args.service == "cohere":
    text_encoder = CohereEmbedding()
else:
    text_encoder = VoyageaiEmbedding()

dataset = MultimodalTextBenchmark(
    root=path,
    name=args.dataset,
    col_to_text_embedder_cfg=TextEmbedderConfig(
        text_embedder=text_encoder,
        batch_size=text_encoder.text_embedder_batch_size,
    ),
)

dataset.materialize(path=osp.join(path, f"data_{args.service}.pt"))

train_dataset, val_dataset, test_dataset = dataset.split()
if len(val_dataset) == 0:
    train_dataset, val_dataset = train_dataset[:0.9], train_dataset[0.9:]

# Set up data loaders
train_loader = DataLoader(train_dataset.tensor_frame,
                          batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset.tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=args.batch_size)

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.embedding: LinearEmbeddingEncoder(),
}

model = FTTransformer(
    channels=args.channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=train_dataset.tensor_frame.col_names_dict,
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
        loss = F.cross_entropy(pred, tf.y)
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
        pred_class = pred.argmax(dim=-1)
        accum += float((tf.y == pred_class).sum())
        total_count += len(tf.y)

    accuracy = accum / total_count
    return accuracy


metric = "Acc"
best_val_metric = 0
best_test_metric = 0

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
          f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}")

print(f"Best Val {metric}: {best_val_metric:.4f}, "
      f"Best Test {metric}: {best_test_metric:.4f}")
