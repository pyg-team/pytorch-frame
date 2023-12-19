import argparse
import os
import os.path as osp
from typing import List, Tuple

import pandas as pd
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
    LinearBucketEncoder,
    LinearEncoder,
)
from torch_frame.typing import TextTokenizationOutputs
from torch_frame.utils import infer_df_stype

# Text Embedded (No pre_transform)
# all-distilroberta-v1
# ================= ResNet ==================
# Best Val Acc: 0.8771, Best Test Acc: 0.8774
# ============= FTTransformer ===============
# Best Val Acc: 0.8736, Best Test Acc: 0.8714

# Text Embedded (thefuzz pre_transform)
# all-distilroberta-v1 + LinearEncoder
# ================= ResNet ==================
# Best Val Acc: 0.8807, Best Test Acc: 0.8798
# ============= FTTransformer ===============
# Best Val Acc: 0.8736, Best Test Acc: 0.8731

# Text Embedded (thefuzz pre_transform)
# all-distilroberta-v1 + LinearBucketEncoder
# ================= ResNet ==================
# Best Val Acc: 0.8795, Best Test Acc: 0.8781
# ============= FTTransformer ===============
# Best Val Acc: 0.8742, Best Test Acc: 0.8760

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--lora", action="store_true")
parser.add_argument("--learning", type=str, default="ResNet",
                    choices=["ResNet", "FTTransformer"])
parser.add_argument("--pre_transform", action="store_true")
parser.add_argument("--linear_bucket", action="store_true")
parser.add_argument(
    "--model",
    type=str,
    default="sentence-transformers/all-distilroberta-v1",
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
    def __init__(self, model: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device)
        self.device = device

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
        return mean_pooling(out.last_hidden_state.detach(), mask).squeeze(1).cpu()


class TextToEmbeddingFinetune(torch.nn.Module):
    r"""Include :obj:`tokenize` that converts text data to tokens, and
    :obj:`forward` function that converts tokens to embeddings with a
    text model, whose parameters will also be finetuned along with the
    tabular learning.

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
        # [batch_size, batch_max_seq_len]
        input_ids = feat["input_ids"].to_dense(fill_value=0).squeeze(dim=1)
        mask = feat["attention_mask"].to_dense(fill_value=0).squeeze(dim=1)

        # Get text embeddings for each text tokenized column
        # `out.last_hidden_state` has the shape:
        # [batch_size, batch_max_seq_len, text_model_out_channels]
        out = self.model(input_ids=input_ids, attention_mask=mask)

        # Return value has the shape [batch_size, 1, text_model_out_channels]
        return mean_pooling(out.last_hidden_state, mask)

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
    text_encoder = TextToEmbedding(model=args.model, device=device)
    text_stype = torch_frame.text_embedded
    kwargs = {
        "text_stype":
        text_stype,
        "col_to_text_embedder_cfg":
        TextEmbedderConfig(text_embedder=text_encoder, batch_size=20),
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

if args.pre_transform:

    def pre_transform(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, dict[str, torch_frame.stype]]:
        from thefuzz import fuzz
        new_df = pd.DataFrame(
            dict(
                ratio=df.apply(
                    lambda x: fuzz.ratio(x["question1"], x["question2"]),
                    axis=1),
                partial_ratio=df.apply(
                    lambda x: fuzz.partial_ratio(x["question1"], x["question2"]
                                                 ), axis=1),
                token_sort_ratio=df.apply(
                    lambda x: fuzz.token_sort_ratio(x["question1"], x[
                        "question2"]), axis=1),
                token_set_ratio=df.apply(
                    lambda x: fuzz.token_set_ratio(x["question1"], x[
                        "question2"]), axis=1),
                partial_token_sort_ratio=df.apply(
                    lambda x: fuzz.partial_token_sort_ratio(
                        x["question1"], x["question2"]), axis=1),
            ))
        return new_df, infer_df_stype(new_df)

    kwargs["pre_transform"] = pre_transform

dataset = QuoraQuestionPairs(root=path, **kwargs)

model_name = args.model.replace('/', '')
filename = f"{model_name}_{text_stype.value}_{args.pre_transform}_data.pt"
dataset.materialize(path=osp.join(path, filename))

dataset = dataset.shuffle()
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

if args.pre_transform:
    if args.linear_bucket:
        stype_encoder_dict[torch_frame.numerical] = LinearBucketEncoder()
    else:
        stype_encoder_dict[torch_frame.numerical] = LinearEncoder()

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
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
    print(f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
          f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}")

print(f"Best Val {metric}: {best_val_metric:.4f}, "
      f"Best Test {metric}: {best_test_metric:.4f}")
