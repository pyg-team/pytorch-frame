import pandas as pd
import pytest
import torch.nn as nn

from torch_frame import TaskType, stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.nn.models.mlp import MLP
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.utils.skorch import NeuralNetClassifierPytorchFrame


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
    (TaskType.BINARY_CLASSIFICATION, nn.BCEWithLogitsLoss),
    (TaskType.MULTICLASS_CLASSIFICATION, nn.CrossEntropyLoss),
])
@pytest.mark.parametrize('pass_dataset', [False, True])
def test_skorch_torchframe_dataset(cls, stypes, task_type_and_loss_cls,
                                   pass_dataset: bool):
    task_type, loss_cls = task_type_and_loss_cls
    loss = loss_cls()

    # initialize dataset
    dataset: Dataset = FakeDataset(
        num_rows=30,
        with_nan=True,
        stypes=stypes,
        create_split=True,
        task_type=task_type,
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(8)),
    )
    dataset.materialize()
    train_dataset, val_dataset, test_dataset = dataset.split()

    # convert to dataframe
    if not pass_dataset:
        df = dataset.df
        df_train = pd.concat([train_dataset.df, val_dataset.df])
        X_train, y_train = df_train.drop(
            columns=[dataset.target_col, dataset.split_col]), df_train[
                dataset.target_col]
        df_test = test_dataset.df
        X_test, y_test = df_test.drop(
            columns=[dataset.target_col, dataset.split_col]), df_test[
                dataset.target_col]

        # never use dataset again
        # we assume that only dataframes are available
        del dataset, train_dataset, val_dataset, test_dataset

    if cls == "mlp":
        channels = 8
        out_channels = 1
        num_layers = 3
        model = MLP(
            channels=channels,
            out_channels=out_channels,
            num_layers=num_layers,
            col_stats={},
            col_names_dict={},
            normalization="layer_norm",
        )
    else:
        raise NotImplementedError

    if pass_dataset:
        net = NeuralNetClassifierPytorchFrame(
            module=model,
            criterion=loss,
            max_epochs=2,
            # lr=args.lr,
            # device=device,
            verbose=1,
            batch_size=1,
        )
        net.fit(dataset)
        y_pred = net.predict(test_dataset)
    else:
        net = NeuralNetClassifierPytorchFrame(
            module=model,
            criterion=loss,
            max_epochs=2,
            # lr=args.lr,
            # device=device,
            verbose=1,
            batch_size=1,
        )
        net.fit(X_train, y_train)
        y_pred = net.predict(X_test)
