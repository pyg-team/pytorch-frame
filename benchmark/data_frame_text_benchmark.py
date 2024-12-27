from __future__ import annotations

import argparse
import math
import os
import os.path as osp
import time
from typing import Any

import torch
from peft import LoraConfig
from peft import TaskType as peftTaskType
from peft import get_peft_model
from tenacity import retry, stop_after_attempt, wait_random_exponential
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import torch_frame
from torch_frame.config import ModelConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data import DataLoader, MultiNestedTensor
from torch_frame.datasets import DataFrameTextBenchmark
from torch_frame.gbdt import GBDT, CatBoost, LightGBM, XGBoost
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
    help="The index of the dataset within DataFrameTextBenchmark",
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
        "roberta-large",
        "microsoft/deberta-v3-large",
        "google/electra-large-discriminator",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/average_word_embeddings_glove.6B.300d",
        "sentence-transformers/all-roberta-large-v1",
        "text-embedding-3-large",
    ],
)
parser.add_argument("--finetune", action="store_true")
parser.add_argument(
    "--pos_weight",
    action="store_true",
    help=("Whether to set `pos_weight` in `BCEWithLogitsLoss` "
          "for the binary classification task."),
)
parser.add_argument('--result_path', type=str, default='')
parser.add_argument("--api_key", type=str, default=None)
args = parser.parse_args()

model_out_channels = {
    "distilbert-base-uncased": 768,
    "roberta-large": 1024,
    "microsoft/deberta-v3-large": 1024,
    "google/electra-large-discriminator": 1024,
    "sentence-transformers/all-distilroberta-v1": 768,
}

# Set for a 16 GB GPU
model_batch_size = {
    "distilbert-base-uncased": 128,
    "roberta-large": 16,
    "microsoft/deberta-v3-large": 8,
    "google/electra-large-discriminator": 16,
    "sentence-transformers/all-distilroberta-v1": 128,
}


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
            target_modules = "all-linear"

        peft_config = LoraConfig(
            task_type=peftTaskType.FEATURE_EXTRACTION,
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


class OpenAIEmbedding:
    def __init__(self, model: str, api_key: str):
        # Please run `pip install openai` to install the package
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, sentences: list[str]) -> Tensor:
        from openai import Embedding

        items: list[Embedding] = embeddings_with_backoff(
            self.client, self.model, sentences)
        assert len(items) == len(sentences)
        embeddings = [
            torch.FloatTensor(item.embedding).view(1, -1) for item in items
        ]
        return torch.cat(embeddings, dim=0)


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6))
def embeddings_with_backoff(client: Any, model: str,
                            sentences: list[str]) -> list[Any]:
    return client.embeddings.create(input=sentences, model=model).data


def mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)


def get_stype_encoder_dict(
    text_stype: torch_frame.stype,
    text_encoder: Any,
    train_tensor_frame: torch_frame.TensorFrame,
) -> dict[torch_frame.stype, StypeEncoder]:
    if not args.finetune:
        text_stype_encoder = LinearEmbeddingEncoder()
    else:
        model_cfg = ModelConfig(
            model=text_encoder,
            out_channels=model_out_channels[args.text_model])
        col_to_model_cfg = {
            col_name: model_cfg
            for col_name in train_tensor_frame.col_names_dict[
                torch_frame.text_tokenized]
        }
        text_stype_encoder = LinearModelEncoder(
            col_to_model_cfg=col_to_model_cfg)

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


def main_gbdt(model: GBDT, train_cfg: dict[str, Any]):
    start_time = time.time()
    model.tune(tf_train=train_dataset.tensor_frame,
               tf_val=val_dataset.tensor_frame,
               num_trials=train_cfg["num_trials"])
    val_pred = model.predict(tf_test=val_dataset.tensor_frame)
    val_metric = model.compute_metric(val_dataset.tensor_frame.y, val_pred)
    test_pred = model.predict(tf_test=test_dataset.tensor_frame)
    test_metric = model.compute_metric(test_dataset.tensor_frame.y, test_pred)
    end_time = time.time()
    result_dict = {
        'args': args.__dict__,
        'best_val_metric': val_metric,
        'best_test_metric': test_metric,
        'best_cfg': model.params,
        'total_time': end_time - start_time,
    }
    print(result_dict)
    # Save results
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_dict, args.result_path)


def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(loader, desc=f"Epoch: {epoch}"):
        tf = tf.to(device)
        y = tf.y
        if isinstance(model, Trompt):
            # Trompt uses the layer-wise loss
            pred = model(tf)
            num_layers = pred.size(1)
            # [batch_size * num_layers, num_classes]
            pred = pred.view(-1, out_channels)
            y = tf.y.repeat_interleave(num_layers)
        else:
            pred = model(tf)
        if pred.size(1) == 1:
            pred = pred.view(-1, )
        if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float)
        loss = loss_fun(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(
    model: Module,
    loader: DataLoader,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if isinstance(model, Trompt):
            # [batch_size, num_layers, out_channels]
            # -> [batch_size, out_channels]
            pred = pred.mean(dim=1)
        if dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif dataset.task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()


def main_torch(
    higher_is_better: bool,
    train_cfg: dict[str, Any],
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    lr_scheduler: Any,
    optimizer: Any,
):
    start_time = time.time()
    if higher_is_better:
        best_val_metric = 0
    else:
        best_val_metric = math.inf

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        val_metric = test(model, val_loader)

        if higher_is_better:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader)
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader)
        lr_scheduler.step()
        print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

    end_time = time.time()
    result_dict = {
        'args': args.__dict__,
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'train_cfg': train_cfg,
        'total_time': end_time - start_time,
    }
    print(result_dict)
    # Save results
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_dict, args.result_path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")

    if not args.finetune:
        if args.text_model == "text-embedding-3-large":
            assert isinstance(args.api_key, str)
            text_encoder = OpenAIEmbedding(model=args.text_model,
                                           api_key=args.api_key)
        else:
            text_encoder = TextToEmbedding(model=args.text_model,
                                           device=device)
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

    # TODO (zecheng): Change this to search space
    batch_size = 512
    if args.finetune:
        batch_size = model_batch_size[args.text_model]
        col_stypes = list(dataset.col_to_stype.values())
        n_tokenized = len([
            col_stype for col_stype in col_stypes
            if col_stype == torch_frame.stype.text_tokenized
        ])
        batch_size //= n_tokenized

    train_cfg = dict(
        channels=128,
        num_layers=4,
        base_lr=0.001,
        epochs=50,
        num_prompts=32,
        batch_size=batch_size,
        gamma_rate=0.9,
        num_trials=1,
    )

    text_model_name = args.text_model.replace('/', '')
    filename = (f"{args.task_type}_{args.scale}_{str(args.idx)}_"
                f"{text_model_name}_{text_stype.value}_data.pt")
    # Notice that different tabular model will reuse materialized dataset:
    dataset.materialize(path=osp.join(path, filename))
    train_dataset, val_dataset, test_dataset = dataset.split()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    train_loader = DataLoader(train_tensor_frame,
                              batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_tensor_frame,
                            batch_size=train_cfg["batch_size"])
    test_loader = DataLoader(test_tensor_frame,
                             batch_size=train_cfg["batch_size"])

    if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        if args.pos_weight:
            label_imbalance = sum(train_tensor_frame.y) / len(
                train_tensor_frame.y)
            loss_fun = BCEWithLogitsLoss(pos_weight=1 / label_imbalance)
        else:
            loss_fun = BCEWithLogitsLoss()
        metric_computer = AUROC(task='binary').to(device)
        higher_is_better = True
    elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = dataset.num_classes
        loss_fun = CrossEntropyLoss()
        metric_computer = Accuracy(task='multiclass',
                                   num_classes=dataset.num_classes).to(device)
        higher_is_better = True
    elif dataset.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fun = MSELoss()
        metric_computer = MeanSquaredError(squared=False).to(device)
        higher_is_better = False

    if args.model_type in GBDT_MODELS:
        # TODO: support gbdt models
        gbdt_cls_dict = {
            "XGBoost": XGBoost,
            "CatBoost": CatBoost,
            "LightGBM": LightGBM
        }
        model_cls = gbdt_cls_dict[args.model_type]
        if dataset.task_type.is_classification:
            num_classes = dataset.num_classes
        else:
            num_classes = None
        model = model_cls(task_type=dataset.task_type, num_classes=num_classes)
        main_gbdt(model, train_cfg)
    else:
        if args.model_type == "FTTransformer":
            model_cls = FTTransformer
            model_kwargs = dict(
                channels=train_cfg["channels"],
                num_layers=train_cfg["num_layers"],
                stype_encoder_dict=get_stype_encoder_dict(
                    text_stype, text_encoder, train_tensor_frame))
        elif args.model_type == "ResNet":
            model_cls = ResNet
            model_kwargs = dict(
                channels=train_cfg["channels"],
                num_layers=train_cfg["num_layers"],
                stype_encoder_dict=get_stype_encoder_dict(
                    text_stype, text_encoder, train_tensor_frame))
        else:
            if args.finetune:
                raise ValueError(
                    "Currently Trompt with finetuning is too expensive")
            model_cls = Trompt
            stype_encoder_dicts = []
            for i in range(train_cfg["num_layers"]):
                stype_encoder_dicts.append(
                    get_stype_encoder_dict(text_stype, text_encoder,
                                           train_tensor_frame))
            model_kwargs = dict(channels=train_cfg["channels"],
                                num_layers=train_cfg["num_layers"],
                                num_prompts=train_cfg["num_prompts"],
                                stype_encoder_dicts=stype_encoder_dicts)
        model = model_cls(
            **model_kwargs,
            out_channels=out_channels,
            col_stats=dataset.col_stats,
            col_names_dict=train_tensor_frame.col_names_dict,
        ).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_cfg["base_lr"])
        lr_scheduler = ExponentialLR(optimizer, gamma=train_cfg["gamma_rate"])
        main_torch(higher_is_better, train_cfg, model, train_loader,
                   val_loader, test_loader, lr_scheduler, optimizer)
