from __future__ import annotations

import argparse
import math
import os
import os.path as osp
import time
from typing import Any

import numpy as np
import optuna
import torch
from peft import LoraConfig
from peft import TaskType as peftTaskType
from peft import get_peft_model
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
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument("--finetune", action="store_true")
parser.add_argument('--result_path', type=str, default='')
parser.add_argument('--num_trials', type=int, default=20,
                    help='Number of Optuna-based hyper-parameter tuning.')
parser.add_argument(
    '--num_repeats', type=int, default=5,
    help='Number of repeated training and eval on the best config.')
args = parser.parse_args()

TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]


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
        model_cfg = ModelConfig(model=text_encoder, out_channels=768)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")

if not args.finetune:
    # text_encoder = TextToEmbedding(model=args.text_model, device=device)
    from torch_frame.testing.text_embedder import HashTextEmbedder
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
filename = (f"{args.task_type}_{args.scale}_{str(args.idx)}_"
            f"{text_model_name}_{text_stype.value}_data.pt")
# Notice that different tabular model will reuse materialized dataset:
dataset.materialize(path=osp.join(path, filename))
train_dataset, val_dataset, test_dataset = dataset.split()

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame

if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
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

if args.model_type == "FTTransformer":
    model_cls = FTTransformer
    model_search_space = {
        'channels': [64, 128, 256],
        'num_layers': [4, 6, 8],
    }
    train_search_space = {
        'batch_size': [256, 512],
        'base_lr': [0.0001, 0.001],
        'gamma_rate': [0.9, 0.95, 1.],
    }
elif args.model_type == "ResNet":
    model_cls = ResNet
    model_search_space = {
        'channels': [64, 128, 256],
        'num_layers': [4, 6, 8],
    }
    train_search_space = {
        'batch_size': [256, 512],
        'base_lr': [0.0001, 0.001],
        'gamma_rate': [0.9, 0.95, 1.],
    }
else:
    if args.finetune:
        raise ValueError("Currently Trompt with finetuning is too expensive")
    model_search_space = {
        'channels': [64, 128, 192],
        'num_layers': [4, 6, 8],
        'num_prompts': [64, 128, 192],
    }
    train_search_space = {
        'batch_size': [128, 256],
        'base_lr': [0.01, 0.001],
        'gamma_rate': [0.9, 0.95, 1.],
    }
    if train_tensor_frame.num_cols > 20:
        # Reducing the model size to avoid GPU OOM
        model_search_space['channels'] = [64, 128]
        model_search_space['num_prompts'] = [64, 128]
    elif train_tensor_frame.num_cols > 50:
        model_search_space['channels'] = [64]
        model_search_space['num_prompts'] = [64]
    model_cls = Trompt


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
            pred = model.forward_stacked(tf)
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
        if dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif dataset.task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()


def objective(trial: optuna.trial.Trial) -> float:
    model_cfg = {}
    for name, search_list in model_search_space.items():
        model_cfg[name] = trial.suggest_categorical(name, search_list)
    train_cfg = {}
    for name, search_list in train_search_space.items():
        train_cfg[name] = trial.suggest_categorical(name, search_list)

    best_val_metric, _ = train_and_eval_with_cfg(
        model_cfg=model_cfg, train_cfg=train_cfg, out_channels=out_channels,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict, device=device,
        model_cls=model_cls, trial=trial)
    return best_val_metric


def main_gbdt(model: GBDT):
    print("GBDT")
    import time
    start_time = time.time()
    model.tune(tf_train=train_dataset.tensor_frame,
               tf_val=val_dataset.tensor_frame, num_trials=args.num_trials)
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


def train_and_eval_with_cfg(
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    out_channels: int,
    col_stats: Any,
    col_names_dict: dict[torch_frame.stype, Any],
    device: torch.device,
    model_cls: Any,
    trial: optuna.trial.Trial | None = None,
) -> tuple[float, float]:
    if args.model_type == "Trompt":
        stype_encoder_dicts = []
        if args.finetune:
            raise ValueError(
                "Currently Trompt with finetuning is too expensive")
        for i in range(train_cfg["num_layers"]):
            stype_encoder_dicts.append(
                get_stype_encoder_dict(text_stype, text_encoder,
                                       train_tensor_frame))
        model = model_cls(
            **model_cfg,
            out_channels=out_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dicts=stype_encoder_dicts,
        ).to(device)
    else:
        stype_encoder_dict = get_stype_encoder_dict(text_stype, text_encoder,
                                                    train_tensor_frame)
        model = model_cls(
            **model_cfg,
            out_channels=out_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        ).to(device)

    train_loader = DataLoader(train_tensor_frame,
                              batch_size=train_cfg["batch_size"], shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_tensor_frame,
                            batch_size=train_cfg["batch_size"])
    test_loader = DataLoader(test_tensor_frame,
                             batch_size=train_cfg["batch_size"])

    model.reset_parameters()
    # Use train_cfg to set up training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["base_lr"])
    lr_scheduler = ExponentialLR(optimizer, gamma=train_cfg["gamma_rate"])
    if higher_is_better:
        best_val_metric = 0
    else:
        best_val_metric = math.inf

    for epoch in range(1, args.epochs + 1):
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

        if trial is not None:
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    print(f'Best val: {best_val_metric:.4f}, '
          f'Best test: {best_test_metric:.4f}')
    return best_val_metric, best_test_metric


def main_torch():
    print("Hyper-parameter search via Optuna")
    start_time = time.time()
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize" if higher_is_better else "minimize",
    )
    study.optimize(objective, n_trials=args.num_trials)
    end_time = time.time()
    search_time = end_time - start_time
    print("Hyper-parameter search done. Found the best config.")
    params = study.best_params
    best_train_cfg = {}
    for train_cfg_key in TRAIN_CONFIG_KEYS:
        best_train_cfg[train_cfg_key] = params.pop(train_cfg_key)
    best_model_cfg = params

    print(f"Repeat experiments {args.num_repeats} times with the best train "
          f"config {best_train_cfg} and model config {best_model_cfg}.")

    best_val_metrics = []
    best_test_metrics = []
    for _ in range(args.num_repeats):
        best_val_metric, best_test_metric = train_and_eval_with_cfg(
            best_model_cfg, best_train_cfg, out_channels, dataset.col_stats,
            train_tensor_frame.col_names_dict, device, model_cls)
        best_val_metrics.append(best_val_metric)
        best_test_metrics.append(best_test_metric)
    end_time = time.time()
    final_model_time = (end_time - start_time) / args.num_repeats
    best_val_metrics = np.array(best_val_metrics)
    best_test_metrics = np.array(best_test_metrics)

    result_dict = {
        'args': args.__dict__,
        'best_val_metrics': best_val_metrics,
        'best_test_metrics': best_test_metrics,
        'best_val_metric': best_val_metrics.mean(),
        'best_test_metric': best_test_metrics.mean(),
        'best_train_cfg': best_train_cfg,
        'best_model_cfg': best_model_cfg,
        'search_time': search_time,
        'final_model_time': final_model_time,
        'total_time': search_time + final_model_time,
    }
    print(result_dict)
    # Save results
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_dict, args.result_path)


if __name__ == "__main__":
    if args.model_type in GBDT_MODELS:
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
        main_gbdt(model)
    else:
        main_torch()
