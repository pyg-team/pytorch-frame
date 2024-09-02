import argparse
import os.path as osp
from tqdm import tqdm
import time

import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torch
from torch_frame.data import DataLoader
from torch_frame.nn import TabTransformer
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from sklearn.metrics import roc_auc_score

from torch_frame import TaskType, stype
from torch_frame.datasets import DataFrameBenchmark

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type', type=str, choices=[
        'binary_classification',
    ], default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                    default='small')
parser.add_argument('--idx', type=int, default=0,
                    help='The index of the dataset within DataFrameBenchmark')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--model_type', type=str, default='TabTransformer',
                    choices=['TabTransformer'])
args = parser.parse_args()

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                            scale=args.scale, idx=args.idx)
dataset.materialize()
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()

def train_tabular_model()-> float:
    def roc_auc(y_hat, y):
        r"""Calculate the Area Under the ROC Curve (AUC)
        for the given predictions and true labels.

        Parameters:
        y_hat (array-like): Predicted probabilities or scores.
        y (array-like): True binary labels.

        Returns:
        float: AUC score.
        """
        return roc_auc_score(y, y_hat)

    # TABULAR
    train_df, val_df, test_df, target_col, cat_col_names, num_col_names = (
        train_dataset.df, val_dataset.df, test_dataset.df, dataset.target_col,
        dataset.tensor_frame.col_names_dict[stype.categorical],
        dataset.tensor_frame.col_names_dict[stype.numerical])

    data_config = DataConfig(
        target=[target_col],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=True,
        batch_size=256,
        max_epochs=args.epochs,
    )

    optimizer_config = OptimizerConfig()

    head_config = LinearHeadConfig(
        layers="520-1040", dropout=0.1, initialization="kaiming",
        use_batch_norm=True,
    ).__dict__  # Convert to dict to pass to the model config

    model_config = TabTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
        ff_hidden_multiplier = 0,
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.config.auto_lr_find = False

    start = time.time()
    tabular_model.fit(train=train_df, validation=val_df,)
    end = time.time()
    tabular_train_time = end - start
    return tabular_train_time

def train_frame_model()-> float:
    # FRAME
    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                            shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

    # Set up model and optimizer
    model = TabTransformer(
        channels=32,
        out_channels=dataset.num_classes,
        num_layers=6,
        num_heads=8,
        encoder_pad_size=2,
        attn_dropout=0.1,
        ffn_dropout=0.1,
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
        for tf in loader:
            tf = tf.to(device)
            pred = model(tf)

            all_labels.append(tf.y.cpu())
            all_preds.append(pred[:, 1].detach().cpu())
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        # Compute the overall AUC
        overall_auc = roc_auc_score(all_labels, all_preds)
        return overall_auc


    metric = 'AUC'
    best_val_metric = 0
    best_test_metric = 0
    start = time.time()
    for epoch in range(1, 2):
        train_loss = train(epoch)
        _ = test(train_loader)
        _ = test(val_loader)

    end = time.time()
    frame_train_time = end - start
    return frame_train_time

frame_train_time = train_frame_model()
tabular_train_time = train_tabular_model()
print(f"Frame training time: {frame_train_time}")
print(f"Tabular training time: {tabular_train_time}")