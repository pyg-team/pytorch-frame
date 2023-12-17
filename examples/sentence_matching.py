import argparse
import os
import os.path as osp
from typing import List

import torch
import torch.nn.functional as F
# Please run `pip install peft` to install the package
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from tqdm import tqdm
# Please run `pip install transformers` to install the package
from transformers import AutoModel, AutoTokenizer

import torch_frame
from torch_frame.config import ModelConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data import DataLoader, MultiNestedTensor
from torch_frame.datasets import QuoraQuestionPairs
from torch_frame.nn import (
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearModelEncoder,
    ResNet,
)
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.typing import TextTokenizationOutputs

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--lora", action="store_true")
parser.add_argument("--learning", type=str, default="ResNet",
                    choices=["ResNet", "FTTransformer"])
parser.add_argument(
    "--model",
    type=str,
    default="distilbert-base-uncased",
    choices=[
        "distilbert-base-uncased",
        "sentence-transformers/all-distilroberta-v1",
    ],
)
parser.add_argument("--pooling", type=str, default="mean",
                    choices=["mean", "cls"])
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()


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


class TextToEmbeddingFinetune(torch.nn.Module):
    r"""Include :obj:`tokenize` that converts text data to tokens, and
    :obj:`forward` function that converts tokens to embeddings with a
    text model, whose parameters will also be finetuned along with the
    tabular learning.

    Args:
        model (str): Model name to load by using :obj:`transformers`,
            such as :obj:`distilbert-base-uncased` and
            :obj:`sentence-transformers/all-distilroberta-v1`.
        pooling (str): Pooling strategy to pool context embeddings into
            sentence level embedding. (default: :obj:`'mean'`)
        lora (bool): Whether using LoRA to finetune the text model.
            (default: :obj:`False`)
    """
    def __init__(self, model: str, pooling: str = "mean", lora: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

        if lora:
            if model == "distilbert-base-uncased":
                target_modules = ["ffn.lin1"]
            elif model == "sentence-transformers/all-distilroberta-v1":
                target_modules = ["intermediate.dense"]
            else:
                raise ValueError(f"Model {model} is not specified for "
                                 f"LoRA finetuning.")

            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=32,
                lora_alpha=32,
                inference_mode=False,
                lora_dropout=0.1,
                bias="none",
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, peft_config)
        self.pooling = pooling

    def forward(self, feat: dict[str, MultiNestedTensor]) -> Tensor:
        # [batch_size, batch_max_seq_len]
        input_ids = feat["input_ids"].to_dense(fill_value=0).squeeze(dim=1)
        mask = feat["attention_mask"].to_dense(fill_value=0).squeeze(dim=1)

        # Get text embeddings for each text tokenized column
        # `out.last_hidden_state` has the shape:
        # [batch_size, batch_max_seq_len, text_model_out_channels]
        out = self.model(input_ids=input_ids, attention_mask=mask)

        # Return value has the shape [batch_size, 1, text_model_out_channels]
        if self.pooling == "mean":
            return mean_pooling(out.last_hidden_state, mask)
        elif self.pooling == "cls":
            return out.last_hidden_state[:, 0, :].unsqueeze(1)
        else:
            raise ValueError(f"{self.pooling} is not supported.")

    def tokenize(self, sentences: List[str]) -> TextTokenizationOutputs:
        # Tokenize batches of sentences
        return self.tokenizer(sentences, truncation=True, padding=True,
                              return_tensors='pt')


def mean_pooling(last_hidden_state: Tensor, attention_mask) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)


torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data",
                "quora_question_pairs")
os.makedirs(path, exist_ok=True)

# Prepare text columns
if not args.finetune:
    # text_encoder = TextToEmbedding(model=args.model, pooling=args.pooling,
    #                                device=device)
    text_encoder = HashTextEmbedder(out_channels=100)
    text_stype = torch_frame.text_embedded
    kwargs = {
        "text_stype":
        text_stype,
        "col_to_text_embedder_cfg":
        TextEmbedderConfig(text_embedder=text_encoder, batch_size=10),
    }
else:
    text_encoder = TextToEmbeddingFinetune(model=args.model,
                                           pooling=args.pooling,
                                           lora=args.lora)
    text_stype = torch_frame.text_tokenized
    kwargs = {
        "text_stype":
        text_stype,
        "col_to_text_tokenizer_cfg":
        TextTokenizerConfig(text_tokenizer=text_encoder.tokenize,
                            batch_size=10000),
    }

dataset = QuoraQuestionPairs(root=path, **kwargs)

model_name = args.model.replace('/', '')
filename = f"{model_name}_{text_stype.value}_data.pt"
dataset.materialize(path=osp.join(path, filename))

dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset[0:0.8], dataset[
    0.8:0.9], dataset[0.9:]

# Set up data loaders
train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

if not args.finetune:
    text_stype_encoder = LinearEmbeddingEncoder()
else:
    model_cfg = ModelConfig(model=text_encoder, out_channels=768)
    col_to_model_cfg = {
        col_name: model_cfg
        for col_name in train_tensor_frame.col_names_dict[
            torch_frame.text_tokenized]
    }
    text_stype_encoder = LinearModelEncoder(col_to_model_cfg=col_to_model_cfg)

stype_encoder_dict = {
    text_stype.parent: text_stype_encoder,
}

if args.learning == "FTTransformer":
    Model = FTTransformer
else:
    Model = ResNet

model = Model(
    channels=args.channels,
    out_channels=dataset.num_classes,
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
    best_val_metric = val_metric
    best_test_metric = test_metric
    print(f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
          f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}")

print(f"Best Val {metric}: {best_val_metric:.4f}, "
      f"Best Test {metric}: {best_test_metric:.4f}")
