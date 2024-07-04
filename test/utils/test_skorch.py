from typing import Any

import pandas as pd
import pytest
import torch.nn as nn
import torch.nn.functional as F
from __furure__ import annotations
from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from torch_frame import TaskType, stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.dataset import Dataset
from torch_frame.data.stats import StatType
from torch_frame.datasets.fake import FakeDataset
from torch_frame.nn.models.mlp import MLP
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.utils.skorch import (
    NeuralNetBinaryClassifierPytorchFrame,
    NeuralNetClassifierPytorchFrame,
)


class BCEWithLogitsLossSigmoidSqueeze(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        # float to long
        input = F.sigmoid(input).float().squeeze()
        target = target.float()
        return super().forward(input, target)


@pytest.mark.parametrize('cls', ["mlp"])
@pytest.mark.parametrize(
    'stypes',
    [
        [stype.numerical],
        [stype.categorical],
        # [stype.text_embedded],
        # [stype.numerical, stype.numerical, stype.text_embedded],
    ])
@pytest.mark.parametrize('task_type_and_loss_cls', [
    (TaskType.REGRESSION, nn.MSELoss),
    (TaskType.BINARY_CLASSIFICATION, BCEWithLogitsLossSigmoidSqueeze),
    (TaskType.MULTICLASS_CLASSIFICATION, nn.CrossEntropyLoss),
])
@pytest.mark.parametrize('pass_dataset', [False])
def test_skorch_torchframe_dataset(cls, stypes, task_type_and_loss_cls,
                                   pass_dataset: bool):
    task_type, loss_cls = task_type_and_loss_cls
    loss = loss_cls()

    # initialize dataset
    dataset: Dataset = FakeDataset(
        num_rows=30,
        # with_nan=True,
        stypes=stypes,
        create_split=True,
        task_type=task_type,
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(8)),
    )
    dataset.materialize()
    train_dataset, val_dataset, test_dataset = dataset.split()
    # print(dataset.col_stats)
    # # convert to dataframe
    # col_to_stype = dataset.col_to_stype
    # # remove split_col and target_col
    # col_to_stype = {
    #     k: v
    #     for k, v in col_to_stype.items()
    #     if k not in [dataset.split_col, dataset.target_col]
    # }
    if not pass_dataset:
        df_train = pd.concat([train_dataset.df, val_dataset.df])
        X_train, y_train = df_train.drop(
            columns=[dataset.target_col, dataset.split_col]), df_train[
                dataset.target_col]
        df_test = test_dataset.df
        X_test, _ = df_test.drop(
            columns=[dataset.target_col, dataset.split_col]), df_test[
                dataset.target_col]

        # never use dataset again
        # we assume that only dataframes are available
        del train_dataset, val_dataset, test_dataset

    if cls == "mlp":

        def get_module(*, col_stats: dict[str, dict[StatType, Any]],
                       col_names_dict: dict[stype, list[str]]) -> MLP:
            channels = 8
            out_channels = 1
            if task_type == TaskType.MULTICLASS_CLASSIFICATION:
                out_channels = dataset.num_classes
            num_layers = 3
            return MLP(
                channels=channels,
                out_channels=out_channels,
                num_layers=num_layers,
                col_stats=col_stats,
                col_names_dict=col_names_dict,
                normalization="layer_norm",
            )
    else:
        raise NotImplementedError

    if task_type in [TaskType.REGRESSION, TaskType.MULTICLASS_CLASSIFICATION]:
        net = NeuralNetClassifierPytorchFrame(
            module=get_module,
            criterion=loss,
            max_epochs=2,
            # lr=args.lr,
            # device=device,
            verbose=1,
            batch_size=3,
            # col_to_stype=col_to_stype,
        )
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        net = NeuralNetBinaryClassifierPytorchFrame(
            module=get_module,
            criterion=loss,
            max_epochs=2,
            # lr=args.lr,
            # device=device,
            verbose=1,
            batch_size=3,
            # col_to_stype=col_to_stype,
        )

    if pass_dataset:
        net.fit(dataset)
        _ = net.predict(test_dataset)
    else:
        net.fit(X_train, y_train)
        _ = net.predict(X_test)


@pytest.mark.parametrize(
    'task_type', [TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION])
def test_sklearn_only(task_type) -> None:
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        X, y = load_iris(return_X_y=True, as_frame=True)
        num_classes = 3
    else:
        X, y = load_diabetes(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    def get_module(*, col_stats: dict[str, dict[StatType, Any]],
                   col_names_dict: dict[stype, list[str]]) -> MLP:
        channels = 8
        out_channels = 1
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            out_channels = num_classes
        num_layers = 3
        return MLP(
            channels=channels,
            out_channels=out_channels,
            num_layers=num_layers,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            normalization="layer_norm",
        )

    net = NeuralNetClassifierPytorchFrame(
        module=get_module,
        criterion=nn.CrossEntropyLoss()
        if task_type == TaskType.MULTICLASS_CLASSIFICATION else nn.MSELoss(),
        max_epochs=2,
        verbose=1,
        lr=0.0001,
        batch_size=3,
    )
    net.fit(X_train, y_train)
    y_pred = net.predict(X_test)

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        assert y_pred.shape == (len(y_test), num_classes)
        acc = accuracy_score(y_test, y_pred.argmax(-1))
        print(acc)
    else:
        assert y_pred.shape == (len(y_test), 1)
        mse = mean_squared_error(y_test, y_pred)
        print(mse)
