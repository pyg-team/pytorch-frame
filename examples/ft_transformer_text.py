import argparse
import os
import os.path as osp
from typing import List

import torch
import torch.nn.functional as F
# Please run `pip install -U sentence-transformers` to install the package
from sentence_transformers import SentenceTransformer
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

# Text embedded:
# ============== wine_reviews ===============
# Best Val Acc: 0.7946, Best Test Acc: 0.7878
# ===== product_sentiment_machine_hack ======
# Best Val Acc: 0.9334, Best Test Acc: 0.8814
# ========== data_scientist_salary ==========
# Best Val Acc: 0.5355, Best Test Acc: 0.4582
# ======== jigsaw_unintended_bias100K =======
# Best Val Acc: 0.9543, Best Test Acc: 0.9511

# Text embeded with Cohere's embed-english-v3.0
# ============== wine_reviews ===============
# Best Val Acc: 0.8263

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wine_reviews')
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_source', type=str, default='huggingface',
                    choices=["huggingface", "cohere"])
parser.add_argument('--cohere_api_key', type=str, default=None)
args = parser.parse_args()

cohere_api_key = args.cohere_api_key or os.environ.get('COHERE_API_KEY', None)
if args.embedding_source == 'cohere' and cohere_api_key is None:
    raise ValueError('Cohere API key is not specified.')


class PretrainedTextEncoder:
    dimension: int = 768
    text_embedder_batch_size: int = 5

    def __init__(self, device: torch.device):
        self.model = SentenceTransformer('all-distilroberta-v1', device=device)

    def __call__(self, sentences: List[str]) -> Tensor:
        # Inference on GPU (if available)
        embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                       convert_to_tensor=True)
        # Map back to CPU
        return embeddings.cpu()


class CohereEmbedding:
    dimension: int = 1024
    text_embedder_batch_size: int = 1000

    def __init__(self, model: str = 'embed-english-v3.0'):
        import cohere  # noqa
        self.model = model
        self.co = cohere.Client(cohere_api_key)

    def __call__(self, sentences: List[str]) -> Tensor:
        items = self.co.embed(model=self.model, texts=sentences,
                              input_type="classification")
        assert len(items) == len(sentences)
        embeddings = torch.tensor(items.embeddings)
        return embeddings


torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
text_encoder = (CohereEmbedding() if args.embedding_source == 'cohere' else
                PretrainedTextEncoder(device=device))

dataset = MultimodalTextBenchmark(
    root=path,
    name=args.dataset,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=text_encoder,
        batch_size=text_encoder.text_embedder_batch_size),
)

dataset.materialize(path=osp.join(path, 'data.pt'))

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
    stype.text_embedded:
    LinearEmbeddingEncoder(in_channels=text_encoder.dimension)
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

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
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
