"""This script benchmarks the training time of TabTransformer using PyTorch
Frame and PyTorch Tabular.

Results form comparing Pytorch Tabular and Frame. Specifically the iteration
speed while trainig.

-------------------------------------
Package | Model   | Num iters/sec|
-------------------------------------
Tabular | TabNet  | 41.7
Frame   | TabNet  | 45.0
Tabular | FTTrans | 40.1
Frame   | FTTrans | 43.7
--------------------------------------
"""
import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import (
    FTTransformerConfig,
    LinearHeadConfig,
    TabTransformerConfig,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from torch_frame import TaskType, stype
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.nn import FTTransformer, TabTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, choices=['binary_classification'],
                    default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                    default='small')
parser.add_argument('--idx', type=int, default=0,
                    help='The index of the dataset within DataFrameBenchmark')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--model_type', type=str, default='TabTransformer',
                    choices=['TabTransformer', 'FTTransformer'])
args = parser.parse_args()

# Data, model params, device setup are the same for both models
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = 32
num_layers = 6
num_heads = 8
encoder_pad_size = 2
attn_dropout = 0.1
ffn_dropout = 0.1
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                             scale=args.scale, idx=args.idx)
dataset.materialize()
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()


def train_tabular_model() -> float:
    """Trains a tabular model with PyTorch Tabular.

    Returns:
        The training time in seconds.
    """
    train_df, val_df, _, target_col, cat_col_names, num_col_names = (
        train_dataset.df, val_dataset.df, test_dataset.df, dataset.target_col,
        dataset.tensor_frame.col_names_dict[stype.categorical],
        dataset.tensor_frame.col_names_dict[stype.numerical])

    data_config = DataConfig(
        target=[target_col],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
    )
    optimizer_config = OptimizerConfig()

    if args.model_type == 'TabTransformer':
        head_config = LinearHeadConfig(
            layers="520-1040",
            dropout=0.1,
            initialization="kaiming",
            use_batch_norm=True,
        ).__dict__  # Convert to dict to pass to the model config
        model_config = TabTransformerConfig(
            task="classification",
            learning_rate=1e-3,
            head="LinearHead",  # Linear Head
            input_embed_dim=channels,
            num_heads=num_heads,
            num_attn_blocks=num_layers,
            attn_dropout=attn_dropout,
            ff_dropout=ffn_dropout,
            head_config=head_config,  # Linear Head Config
            ff_hidden_multiplier=0,
        )
    elif args.model_type == 'FTTransformer':
        head_config = LinearHeadConfig(
            layers=f"{channels}-{dataset.num_classes}",
            dropout=0.1,
            initialization="kaiming",
            use_batch_norm=True,
        ).__dict__  # Convert to dict to pass to the model config
        model_config = FTTransformerConfig(
            task="classification",
            learning_rate=1e-3,
            head="LinearHead",  # Linear Head
            input_embed_dim=channels,
            # dividing by 4 to match the number of params
            # in FTTransformer from torch frame
            num_heads=int(num_heads / 4),
            num_attn_blocks=num_layers,
            attn_dropout=attn_dropout,
            head_config=head_config,  # Linear Head Config
            ff_hidden_multiplier=0,
        )
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    start = time.time()
    tabular_model.fit(
        train=train_df,
        validation=val_df,
    )
    return time.time() - start


def train_frame_model() -> float:
    """Trains a tabular model with PyTorch Frame.

    Returns:
        The training time in seconds.
    """
    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    train_loader = DataLoader(
        train_tensor_frame,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
    # Set up model and optimizer
    if args.model_type == 'TabTransformer':
        model = TabTransformer(
            channels=channels,
            out_channels=dataset.num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            encoder_pad_size=2,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            col_stats=dataset.col_stats,
            col_names_dict=train_tensor_frame.col_names_dict,
        ).to(device)
    elif args.model_type == 'FTTransformer':
        model = FTTransformer(
            channels=channels,
            out_channels=dataset.num_classes,
            num_layers=num_layers,
            col_stats=dataset.col_stats,
            col_names_dict=train_tensor_frame.col_names_dict,
        ).to(device)

    num_params = 0
    for m in model.parameters():
        if m.requires_grad:
            num_params += m.numel()
    print(f'Number of parameters: {num_params}')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(epoch: int) -> float:
        model.train()
        loss_accum = total_count = 0
        for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
            tf = tf.to(device)
            pred = model.forward(tf)
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
        all_preds = []
        all_labels = []
        for tf in tqdm(loader):
            tf = tf.to(device)
            pred = model(tf)
            all_labels.append(tf.y.cpu())
            all_preds.append(pred[:, 1].cpu())
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        return roc_auc_score(all_labels, all_preds)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        _ = train(epoch)
        _ = test(val_loader)

    return time.time() - start


frame_train_time = train_frame_model()
tabular_train_time = train_tabular_model()
print(f"Model type: {args.model_type}. Device: {device}")
print(f"Frame average time per epoch: "
      f"{frame_train_time / args.epochs:.2f}s")
print(f"Tabular average time per epoch: "
      f"{tabular_train_time / args.epochs:.2f}s")
