from skorch import NeuralNetClassifier, NeuralNet
from skorch.dataset import Dataset as SkorchDataset
import torch.nn as nn
import torch_frame
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.utils import infer_df_stype
from torch_frame.data.dataset import DataFrameToTensorFrameConverter, Dataset
from torch_frame.data.loader import DataLoader
import torch
from torch_frame.typing import IndexSelectType
from torch import Tensor
from pandas import DataFrame
from typing import Any
import pandas as pd
from numpy.typing import ArrayLike
from torch_frame.config import (
    ImageEmbedderConfig,
    TextEmbedderConfig,
    TextTokenizerConfig,
)

class NeuralNetPytorchFrameDataLoader(DataLoader):
    def __init__(
        self, dataset: Dataset | TensorFrame, *args, device: torch.device, **kwargs
    ):
        super().__init__(dataset, *args, **kwargs)
        self.device = device

    def collate_fn(self, index: IndexSelectType) -> tuple[TensorFrame, Tensor | None]:
        index = torch.tensor(index)
        res = super().collate_fn(index).to(self.device)
        return res, res.y


class NeuralNetPytorchFrame(NeuralNet):
    def __init__(
        self,
        # NeuralNet parameters
        module,
        criterion,
        optimizer=torch.optim.SGD,
        lr=0.01,
        max_epochs=10,
        batch_size=128,
        iterator_train=...,
        iterator_valid=...,
        dataset=...,
        train_split=...,
        callbacks=None,
        predict_nonlinearity="auto",
        warm_start=False,
        verbose=1,
        device="cpu",
        compile=False,
        use_caching="auto",
        # torch_frame.Dataset parameters
        col_to_stype: dict[str, torch_frame.stype] | None = None,
        target_col: str | None = None,
        split_col: str | None = None,
        col_to_sep: str | None | dict[str, str | None] = None,
        col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
        | TextEmbedderConfig | None = None,
        col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
        | TextTokenizerConfig | None = None,
        col_to_image_embedder_cfg: dict[str, ImageEmbedderConfig]
        | ImageEmbedderConfig | None = None,
        col_to_time_format: str | None | dict[str, str | None] = None,
        # other NeuralNet parameters
        **kwargs,
    ):
        super().__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            iterator_train=self.iterator_train_valid,  # changed
            iterator_valid=self.iterator_train_valid,  # changed
            dataset=self.create_dataset,  # changed
            train_split=self.split_dataset,  # changed
            callbacks=callbacks,
            predict_nonlinearity=predict_nonlinearity,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            compile=compile,
            use_caching=use_caching,
            **kwargs,
        )
        self.col_to_stype = col_to_stype
        self.target_col = target_col
        self.split_col = split_col
        self.col_to_sep = col_to_sep
        self.col_to_text_embedder_cfg = col_to_text_embedder_cfg
        self.col_to_text_tokenizer_cfg = col_to_text_tokenizer_cfg
        self.col_to_image_embedder_cfg = col_to_image_embedder_cfg
        self.col_to_time_format = col_to_time_format

    def create_dataset(self, df: DataFrame, _: Any) -> Dataset:
        dataset_ = Dataset(
            df,
            self.dataset_.col_to_stype,
            split_col=self.dataset_.split_col,
            target_col=self.dataset_.target_col,
            col_to_sep=self.dataset_.col_to_sep,
            col_to_text_embedder_cfg=self.dataset_.col_to_text_embedder_cfg,
            col_to_text_tokenizer_cfg=self.dataset_.col_to_text_tokenizer_cfg,
            col_to_image_embedder_cfg=self.dataset_.col_to_image_embedder_cfg,
            col_to_time_format=self.dataset_.col_to_time_format,
        )
        dataset_.materialize()
        return dataset_

    def split_dataset(self, dataset: Dataset) -> tuple[TensorFrame, TensorFrame]:
        datasets = dataset.split()[:2]
        return datasets[0].tensor_frame, datasets[1].tensor_frame

    def iterator_train_valid(self, dataset: Dataset, **kwargs: Any) -> DataLoader:
        return NeuralNetPytorchFrameDataLoader(dataset, device=self.device, **kwargs)

    def fit(self, X: Dataset | DataFrame, y: ArrayLike | None=None, **fit_params):
        if isinstance(X, DataFrame):
            if y is not None:
                X["target_col"] = y
            self.dataset_ = Dataset(
                X,
                self.col_to_stype or infer_df_stype(X),
                split_col=self.split_col,
                target_col=self.target_col,
                col_to_sep=self.col_to_sep,
                col_to_text_embedder_cfg=self.col_to_text_embedder_cfg,
                col_to_text_tokenizer_cfg=self.col_to_text_tokenizer_cfg,
                col_to_image_embedder_cfg=self.col_to_image_embedder_cfg,
                col_to_time_format=self.col_to_time_format,
            )
        else:
            self.dataset_ = X
        return super().fit(self.dataset_.df, None, **fit_params)

# TODO: make this behave more like NeuralNetClassifier
class NeuralNetClassifierPytorchFrame(NeuralNetPytorchFrame):
    def fit(self, X: Dataset | DataFrame, y: ArrayLike | None=None, **fit_params):
        fit_result = super().fit(X, y, **fit_params)
        self.classes = self.dataset_.df["target_col"].unique()
        return fit_result