import argparse
import os.path as osp
from typing import List

import openai
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import DataLoader
from torch_frame.datasets import AmazonFineFoodReviews
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
)

# Run `export OPENAI_API_KEY=...` before running the code
# Notice that there are 568,454 rows and 2 text columns, it will
# cost some money to get the text embeddings by using OpenAI API


class OpenAIEmbedding:
    def __init__(self, model: str = 'text-embedding-ada-002'):
        self.model = model

    def __call__(self, sentences: List[str]) -> Tensor:
        items = openai.Embedding.create(input=sentences,
                                        model=self.model)['data']
        assert len(items) == len(sentences)
        embeddings = [
            torch.FloatTensor(item['embedding']).view(1, -1) for item in items
        ]
        return torch.cat(embeddings, dim=0)


parser = argparse.ArgumentParser()
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='text-embedding-ada-002')
parser.add_argument('--emb_size', type=int, default=1536)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'amazon_fine_food_reviews')
text_encoder = OpenAIEmbedding(model=args.model)
dataset = AmazonFineFoodReviews(
    root=path,
    text_embedder_cfg=TextEmbedderConfig(text_embedder=text_encoder,
                                         batch_size=5),
)

dataset.materialize(path=osp.join(path, 'data.pt'))

# Shuffle the dataset
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
    stype.text_embedded: LinearEmbeddingEncoder(in_channels=args.emb_size)
}

output_channels = dataset.num_classes

model = FTTransformer(
    channels=args.channels,
    out_channels=output_channels,
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


metric = 'Acc'
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

    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
          f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')

print(f'Best Val {metric}: {best_val_metric:.4f}, '
      f'Best Test {metric}: {best_test_metric:.4f}')
