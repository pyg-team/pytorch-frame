from __future__ import annotations

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import torch_frame
from torch_frame.config import ModelConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data import DataLoader, MultiNestedTensor
from torch_frame.datasets import DataFrameTextBenchmark
from torch_frame.gbdt import CatBoost, LightGBM, XGBoost
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
    LinearModelEncoder,
    MultiCategoricalEmbeddingEncoder,
    ResNet,
    StypeEncoder,
    Trompt,
)
from torch_frame.nn.encoder.stype_encoder import TimestampEncoder
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.typing import TaskType, TextTokenizationOutputs

GBDT_MODELS = ["XGBoost", "CatBoost", "LightGBM"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_type",
    type=str,
    choices=[
        "binary_classification", "multiclass_classification", "regression"
    ],
    default="binary_classification",
)
parser.add_argument(
    "--scale",
    type=str,
    choices=["small", "medium", "large"],
    default="small",
)
parser.add_argument(
    "--idx",
    type=int,
    default=0,
    help="The index of the dataset within DataFrameBenchmark",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="FTTransformer",
    choices=[
        "FTTransformer", "ResNet", "Trompt", "XGBoost", "CatBoost", "LightGBM"
    ],
)
parser.add_argument(
    "--text_model",
    type=str,
    default="sentence-transformers/all-distilroberta-v1",
    choices=[
        "distilbert-base-uncased",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/average_word_embeddings_glove.6B.300d",
    ],
)
parser.add_argument("--finetune", action="store_true")
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
    """
    def __init__(self, model: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

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
                              return_tensors="pt")


def mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)


def get_stype_encoder_dict(text_stype: torch_frame.stype,
                           text_stype_encoder: StypeEncoder) -> StypeEncoder:
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
    return stype_encoder_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")

# Hyperparameters
channels = 128
num_layers = 4
lr = 0.001
epochs = 20
num_prompts = 20

if not args.finetune:
    # text_encoder = TextToEmbedding(model=args.text_model, device=device)
    text_encoder = HashTextEmbedder(100)
    text_stype = torch_frame.text_embedded
    kwargs = {
        "text_stype":
        text_stype,
        "col_to_text_embedder_cfg":
        TextEmbedderConfig(text_embedder=text_encoder, batch_size=5),
    }
else:
    text_encoder = TextToEmbeddingFinetune(model=args.text_model)
    text_stype = torch_frame.text_tokenized
    kwargs = {
        "text_stype":
        text_stype,
        "col_to_text_tokenizer_cfg":
        TextTokenizerConfig(text_tokenizer=text_encoder.tokenize,
                            batch_size=10000),
    }

dataset = DataFrameTextBenchmark(
    root=path,
    task_type=TaskType(args.task_type),
    scale=args.scale,
    idx=args.idx,
    **kwargs,
)

text_model_name = args.text_model.replace('/', '')
filename = (f"{text_model_name}_{args.task_type}_{args.scale}_"
            f"{str(args.idx)}_{text_stype.value}_data.pt")
dataset.materialize(path=osp.join(path, filename))
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=512, shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=512)
test_loader = DataLoader(test_tensor_frame, batch_size=512)

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

is_classification = dataset.task_type.is_classification
if is_classification:
    out_channels = dataset.num_classes
else:
    out_channels = 1

if args.model_type in GBDT_MODELS:
    # TODO: support gbdt models
    raise NotImplementedError
    gbdt_cls_dict = {
        "XGBoost": XGBoost,
        "CatBoost": CatBoost,
        "LightGBM": LightGBM
    }
    model_cls = gbdt_cls_dict[args.model_type]
elif args.model_type == "FTTransformer":
    model_cls = FTTransformer
    model_cfg = dict(
        channels=channels, num_layers=num_layers,
        stype_encoder_dict=get_stype_encoder_dict(text_stype,
                                                  text_stype_encoder))
elif args.model_type == "ResNet":
    model_cls = ResNet
    model_cfg = dict(
        channels=channels, num_layers=num_layers,
        stype_encoder_dict=get_stype_encoder_dict(text_stype,
                                                  text_stype_encoder))
else:
    if args.finetune:
        raise ValueError("Currently Trompt with finetuning is too expensive")
    model_cls = Trompt
    stype_encoder_dicts = []
    for i in range(num_layers):
        stype_encoder_dicts.append(
            get_stype_encoder_dict(text_stype, text_stype_encoder))
    model_cfg = dict(channels=channels, num_layers=num_layers,
                     num_prompts=num_prompts,
                     stype_encoder_dicts=stype_encoder_dicts)

model = model_cls(
    **model_cfg,
    out_channels=out_channels,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
).to(device)
model.reset_parameters()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


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

for epoch in range(1, epochs + 1):
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
