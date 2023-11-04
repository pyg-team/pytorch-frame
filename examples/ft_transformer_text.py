import argparse
import os.path as osp
from typing import Dict, List

import torch
import torch.nn.functional as F
# Please run `pip install -U sentence-transformers` to install the package
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
    LinearEmbeddingModelEncoder,
    LinearEncoder,
)
from torch_frame.typing import TensorData

# Text embedded:
# ============== wine_reviews ===============
# Best Val Acc: 0.7946, Best Test Acc: 0.7878
# ===== product_sentiment_machine_hack ======
# Best Val Acc: 0.9334, Best Test Acc: 0.8814
# ========== data_scientist_salary ==========
# Best Val Acc: 0.5355, Best Test Acc: 0.4582
# ======== jigsaw_unintended_bias100K =======
# Best Val Acc: 0.9543, Best Test Acc: 0.9511


class PretrainedTextEncoder:
    def __init__(self, device: torch.device):
        self.model = SentenceTransformer('all-distilroberta-v1', device=device)

    def __call__(self, sentences: List[str]) -> Tensor:
        # Inference on GPU (if available)
        embeddings = self.model.encode(sentences, convert_to_numpy=False,
                                       convert_to_tensor=True)
        # Map back to CPU
        return embeddings.cpu()


class TextTokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, sentences: List[str]) -> List[Dict[str, Tensor]]:
        res = []
        for s in sentences:
            inputs = self.tokenizer(s, return_tensors='pt')
            res.append(inputs)
        return res


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str, pooling: str = 'mean'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, feat: TensorData) -> Tensor:
        input_ids = feat['input_ids'].to_dense(fill_value=0)
        attention_mask = feat['attention_mask'].to_dense(fill_value=0)
        outs = []
        for i in range(input_ids.shape[1]):
            out = self.model(input_ids=input_ids[:, i, :],
                             attention_mask=attention_mask[:, i, :])
            if self.pooling == 'mean':
                outs.append(
                    self.mean_pooling(out.last_hidden_state,
                                      attention_mask[:, i, :]))
            elif self.pooling == 'cls':
                outs.append(out.last_hidden_state[:, 0, :])
            else:
                raise ValueError(f'{self.pooling} is not supported.')
        return torch.cat(outs, dim=1)

    def mean_pooling(self, last_hidden_state: Tensor,
                     attention_mask) -> Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()
        embedding = torch.sum(
            last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
        return embedding.unsqueeze(1)

    def reset_parameters(self):
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wine_reviews')
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--text_model', type=str,
                    default='distilbert-base-uncased',
                    choices=['distilbert-base-uncased'])
parser.add_argument('--pooling', type=str, default='mean',
                    choices=['mean', 'cls'])
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)

# Prepare text columns
if not args.finetune:
    text_encoder = PretrainedTextEncoder(device=device)
    text_stype = torch_frame.text_embedded
    text_stype_encoder = LinearEmbeddingEncoder(in_channels=768)
    kwargs = {
        'text_stype':
        text_stype,
        'text_embedder_cfg':
        TextEmbedderConfig(text_embedder=text_encoder, batch_size=5),
    }
else:
    text_tokenizer = TextTokenizer(model_name=args.text_model)
    text_encoder = TextEncoder(model_name=args.text_model,
                               pooling=args.pooling)
    text_stype = torch_frame.text_tokenized
    text_stype_encoder = LinearEmbeddingModelEncoder(in_channels=768,
                                                     model=text_encoder)
    kwargs = {
        'text_stype':
        text_stype,
        'text_tokenizer_cfg':
        TextTokenizerConfig(text_tokenizer=text_tokenizer, batch_size=10000),
    }

dataset = MultimodalTextBenchmark(root=path, name=args.dataset, **kwargs)

dataset.materialize(path=osp.join(path, 'data.pt'))

print(dataset._tensor_frame.feat_dict[torch_frame.text_tokenized])

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
