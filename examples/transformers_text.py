from __future__ import annotations

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
# Please run `pip install peft` to install the package
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from tqdm import tqdm
# Please run `pip install transformers` to install the package
from transformers import AutoModel, AutoTokenizer

import torch_frame
from torch_frame import stype
from torch_frame.config import ModelConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data import DataLoader, MultiNestedTensor
from torch_frame.datasets import MultimodalTextBenchmark
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
    LinearModelEncoder,
    MultiCategoricalEmbeddingEncoder,
)
from torch_frame.nn.encoder.stype_encoder import TimestampEncoder
from torch_frame.typing import TextTokenizationOutputs

# Text Embedded
# all-distilroberta-v1
# ============== wine_reviews ===============
# Best Val Acc: 0.7968, Best Test Acc: 0.7926
# ===== product_sentiment_machine_hack ======
# Best Val Acc: 0.9155, Best Test Acc: 0.8885
# ======== jigsaw_unintended_bias100K =======
# Best Val Acc: 0.9470, Best Test Acc: 0.9488
# =============== news_channel ==============
# Best Val Acc: 0.5010, Best Test Acc: 0.4847
# ============ fake_job_postings2 ===========
# Best Val Acc: 0.9788, Best Test Acc: 0.9739
# ========== imdb_genre_prediction ==========
# Best Val Acc: 0.7875, Best Test Acc: 0.6900

# e5-mistral-7b-instruct
# ============== wine_reviews ===============
# Best Val Acc: 0.8289, Best Test Acc: 0.8203
# ===== product_sentiment_machine_hack ======
# Best Val Acc: 0.9273, Best Test Acc: 0.9159
# ======== jigsaw_unintended_bias100K =======
# Best Val Acc: 0.9571, Best Test Acc: 0.9566
# =============== news_channel ==============
# Best Val Acc: 0.4837, Best Test Acc: 0.4711
# ============ fake_job_postings2 ===========
# Best Val Acc: 0.9851, Best Test Acc: 0.9818
# ========== imdb_genre_prediction ==========
# Best Val Acc: 0.9500, Best Test Acc: 0.8300

# Text Tokenized
# distilbert-base-uncased + LoRA
# ============== wine_reviews ===============
# Best Val Acc: 0.8314, Best Test Acc: 0.8230
# ===== product_sentiment_machine_hack ======
# Best Val Acc: 0.9096, Best Test Acc: 0.8908
# ======== jigsaw_unintended_bias100K =======
# Best Val Acc: 0.9672, Best Test Acc: 0.9644
# =============== news_channel ==============
# Best Val Acc: 0.5039, Best Test Acc: 0.4918
# ============ fake_job_postings2 ===========
# Best Val Acc: 0.9788, Best Test Acc: 0.9736
# ========== imdb_genre_prediction ==========
# Best Val Acc: 0.8125, Best Test Acc: 0.7150

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="wine_reviews")
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--lora", action="store_true")
parser.add_argument(
    "--model",
    type=str,
    default="distilbert-base-uncased",
    choices=[
        "distilbert-base-uncased",
        "sentence-transformers/all-distilroberta-v1",
        "intfloat/e5-mistral-7b-instruct",
    ],
)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

if args.lora and not args.finetune:
    raise ValueError("Please also specify finetune when "
                     "choosing LoRA finetuning.")


class TextToEmbedding:
    def __init__(self, model: str, device: torch.device):
        self.model_name = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if model == "intfloat/e5-mistral-7b-instruct":
            # Use last pooling here because this model is
            # a decoder (causal) language model that only
            # the last token attends to all previous tokens:
            self.pooling = "last"
            self.model = AutoModel.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
            ).to(device)
        else:
            self.model = AutoModel.from_pretrained(model).to(device)
            self.pooling = "mean"

    def __call__(self, sentences: list[str]) -> Tensor:
        if self.model_name == "intfloat/e5-mistral-7b-instruct":
            sentences = [(f"Instruct: Retrieve relevant knowledge and "
                          f"embeddings.\nQuery: {sentence}")
                         for sentence in sentences]
            max_length = 4096
            inputs = self.tokenizer(
                sentences,
                max_length=max_length - 1,
                truncation=True,
                return_attention_mask=False,
                padding=False,
            )
            inputs["input_ids"] = [
                input_ids + [self.tokenizer.eos_token_id]
                for input_ids in inputs["input_ids"]
            ]
            inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
        else:
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

        if self.pooling == "mean":
            return (mean_pooling(out.last_hidden_state.detach(),
                                 mask).squeeze(1).cpu())
        elif self.pooling == "last":
            return last_pooling(out.last_hidden_state,
                                mask).detach().cpu().to(torch.float32)
        else:
            raise ValueError(f"{self.pooling} is not supported.")


class TextToEmbeddingFinetune(torch.nn.Module):
    r"""Include :obj:`tokenize` that converts text data to tokens, and
    :obj:`forward` function that converts tokens to embeddings with a
    text model, whose parameters will also be finetuned along with the
    tabular learning. The pooling strategy used here to derive sentence
    embedding is the mean pooling which takes mean value of all tokens'
    embeddings.

    Args:
        model (str): Model name to load by using :obj:`transformers`,
            such as :obj:`distilbert-base-uncased` and
            :obj:`sentence-transformers/all-distilroberta-v1`.
        lora (bool): Whether using LoRA to finetune the text model.
            (default: :obj:`False`)
    """
    def __init__(self, model: str, lora: bool = False):
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

    def forward(self, feat: dict[str, MultiNestedTensor]) -> Tensor:
        # Pad [batch_size, 1, *] into [batch_size, 1, batch_max_seq_len], then,
        # squeeze to [batch_size, batch_max_seq_len].
        input_ids = feat["input_ids"].to_dense(fill_value=0).squeeze(dim=1)
        # Set attention_mask of padding idx to be False
        mask = feat["attention_mask"].to_dense(fill_value=0).squeeze(dim=1)

        # Get text embeddings for each text tokenized column
        # `out.last_hidden_state` has the shape:
        # [batch_size, batch_max_seq_len, text_model_out_channels]
        out = self.model(input_ids=input_ids, attention_mask=mask)

        # Return value has the shape [batch_size, 1, text_model_out_channels]
        return mean_pooling(out.last_hidden_state, mask)

    def tokenize(self, sentences: list[str]) -> TextTokenizationOutputs:
        # Tokenize batches of sentences
        return self.tokenizer(sentences, truncation=True, padding=True,
                              return_tensors='pt')


def mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)


def last_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    # Find the last token that attends to previous tokens.
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths]


torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data",
                args.dataset)

# Prepare text columns
if not args.finetune:
    text_encoder = TextToEmbedding(model=args.model, device=device)
    text_stype = torch_frame.text_embedded
    kwargs = {
        "text_stype":
        text_stype,
        "col_to_text_embedder_cfg":
        TextEmbedderConfig(text_embedder=text_encoder, batch_size=1),
    }
else:
    text_encoder = TextToEmbeddingFinetune(model=args.model, lora=args.lora)
    text_stype = torch_frame.text_tokenized
    kwargs = {
        "text_stype":
        text_stype,
        "col_to_text_tokenizer_cfg":
        TextTokenizerConfig(text_tokenizer=text_encoder.tokenize,
                            batch_size=10000),
    }

dataset = MultimodalTextBenchmark(root=path, name=args.dataset, **kwargs)

model_name = args.model.replace('/', '')
filename = f"{model_name}_{text_stype.value}_data.pt"
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
    stype.categorical:
    EmbeddingEncoder(),
    stype.numerical:
    LinearEncoder(),
    # If text_stype is text_embedded,
    # it becomes embedding after materialization
    text_stype.parent:
    text_stype_encoder,
    stype.multicategorical:
    MultiCategoricalEmbeddingEncoder(),
    stype.timestamp:
    TimestampEncoder()
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
