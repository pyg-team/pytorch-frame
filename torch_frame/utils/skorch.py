from __future__ import annotations

import importlib
from typing import Any
import warnings

import skorch.utils
import torch
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from skorch import NeuralNet
from torch import Tensor

import torch_frame
from torch_frame.config import (
    ImageEmbedderConfig,
    TextEmbedderConfig,
    TextTokenizerConfig,
)
from torch_frame.data.dataset import Dataset
from torch_frame.data.loader import DataLoader
from torch_frame.data.tensor_frame import TensorFrame
from torch_frame.typing import IndexSelectType
from torch_frame.utils import infer_df_stype

# TODO: make it more safe
old_to_tensor = skorch.utils.to_tensor


def to_tensor(X, device, accept_sparse=False):
    if isinstance(X, TensorFrame):
        return X
    return old_to_tensor(X, device, accept_sparse)


skorch.utils.to_tensor = to_tensor

importlib.reload(skorch.net)


class NeuralNetPytorchFrameDataLoader(DataLoader):
    def __init__(self, dataset: Dataset | TensorFrame, *args,
                 device: torch.device, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.device = device

    def collate_fn(  # type: ignore
            self, index: IndexSelectType) -> tuple[TensorFrame, Tensor | None]:
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
        iterator_train=None,
        iterator_valid=None,
        dataset=None,
        train_split=None,
        callbacks=None,
        predict_nonlinearity="auto",
        warm_start=False,
        verbose=1,
        device="cpu",
        compile=False,
        use_caching="auto",
        # torch_frame.Dataset parameters
        col_to_stype: dict[str, torch_frame.stype] | None = None,
        target_col: str | None = "target_col",
        split_col: str | None = "split_col",
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
    ) -> None:
        """`skorch.NeuralNet` with `torch_frame` support.
        
        Additional parameters are **ONLY** used when creating a dummy torch_frame.data.dataset.Dataset
        if pandas.DataFrame is passed as X in `fit` or `predict` methods.

        Parameters
        ----------
        col_to_stype (Dict[str, torch_frame.stype]): A dictionary that maps
            each column in the data frame to a semantic type.
        target_col (str, optional): The column used as target.
            (default: :obj:`None`)
        split_col (str, optional): The column that stores the pre-defined split
            information. The column should only contain :obj:`0`, :obj:`1`, or
            :obj:`2`. (default: :obj:`None`).
        col_to_sep (Union[str, Dict[str, Optional[str]]]): A dictionary or a
            string/:obj:`None` specifying the separator/delimiter for the
            multi-categorical columns. If a string/:obj:`None` is specified,
            then the same separator will be used throughout all the
            multi-categorical columns. Note that if :obj:`None` is specified,
            it assumes a multi-category is given as a :obj:`list` of
            categories. If a dictionary is given, we use a separator specified
            for each column. (default: :obj:`None`)
        col_to_text_embedder_cfg (TextEmbedderConfig or dict, optional):
            A text embedder configuration or a dictionary of configurations
            specifying :obj:`text_embedder` that embeds texts into vectors and
            :obj:`batch_size` that specifies the mini-batch size for
            :obj:`text_embedder`. (default: :obj:`None`)
        col_to_text_tokenizer_cfg (TextTokenizerConfig or dict, optional):
            A text tokenizer configuration or dictionary of configurations
            specifying :obj:`text_tokenizer` that maps sentences into a
            list of dictionary of tensors. Each element in the list
            corresponds to each sentence, keys are input arguments to
            the model such as :obj:`input_ids`, and values are tensors
            such as tokens. :obj:`batch_size` specifies the mini-batch
            size for :obj:`text_tokenizer`. (default: :obj:`None`)
        col_to_image_embedder_cfg (ImageEmbedderConfig or dict, optional):
            No documentation provided.
        col_to_time_format (Union[str, Dict[str, Optional[str]]], optional): A
            dictionary or a string specifying the format for the timestamp
            columns. See `strfttime documentation
            <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`_
            for more information on formats. If a string is specified,
            then the same format will be used throughout all the timestamp
            columns. If a dictionary is given, we use a different format
            specified for each column. If not specified, pandas's internal
            to_datetime function will be used to auto parse time columns.
            (default: :obj:`None`)
        """
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
        # additional parameters used when creating a dummy 
        # torch_frame.data.dataset.Dataset
        self.col_to_stype = col_to_stype
        self.target_col = target_col
        self.split_col = split_col
        self.col_to_sep = col_to_sep
        self.col_to_text_embedder_cfg = col_to_text_embedder_cfg
        self.col_to_text_tokenizer_cfg = col_to_text_tokenizer_cfg
        self.col_to_image_embedder_cfg = col_to_image_embedder_cfg
        self.col_to_time_format = col_to_time_format
        # save dataset for partial_fit
        self.train_split_original = train_split or (
            lambda x: train_test_split(x, test_size=0.2))
        # 0.2 is the default test_size in train_test_split in skorch

    def create_dataset(self, df: DataFrame, _: Any) -> Dataset:
        # skorch API
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

    def split_dataset(self,
                      dataset: Dataset) -> tuple[TensorFrame, TensorFrame]:
        # skorch API
        datasets = dataset.split()[:2]
        return datasets[0].tensor_frame, datasets[1].tensor_frame

    def iterator_train_valid(self, dataset: Dataset,
                             **kwargs: Any) -> DataLoader:
        # skorch API
        return NeuralNetPytorchFrameDataLoader(dataset, device=self.device,
                                               **kwargs)

    def fit(self, X: Dataset | DataFrame, y: ArrayLike | None = None,
            **fit_params):
        if isinstance(X, DataFrame):
            # create target_col if not exists
            if y is not None:
                X[self.target_col] = y
            elif self.target_col not in X:
                warnings.warn(
                    f"target_col {self.target_col} not found in X and y is None",
                    UserWarning
                , stacklevel=2)
                
            # create split_col if not exists
            if self.split_col not in X:
                # first split the data with the split function
                X_train, X_val = self.train_split_original(X, **fit_params)
                # if index is in X_train, 0, otherwise 1
                # X[self.split_col] = (X.index.isin(X_train.index)).astype(int)
                # split_col uses iloc instead of loc, this is weird
                X[self.split_col] = (X.index.isin(X_train.index)).astype(int)
            
            self.dataset_ = Dataset(
                X,
                # do not include split_col
                {  # type: ignore
                    k: v
                    for k, v in infer_df_stype(X).items()
                    if k not in (self.split_col, )
                } | (self.col_to_stype or {}),
                split_col=self.split_col,
                target_col=self.target_col,
                col_to_sep=self.col_to_sep,
                col_to_text_embedder_cfg=self.col_to_text_embedder_cfg,
                col_to_text_tokenizer_cfg=self.col_to_text_tokenizer_cfg,
                col_to_image_embedder_cfg=self.col_to_image_embedder_cfg,
                col_to_time_format=self.col_to_time_format,
            )
            # materialize the dataset to add col_stats and col_names_dict
            self.dataset_.materialize()
        else:
            self.dataset_ = X

        # self.module.encoder.col_stats = self.dataset_.col_stats
        # self.module.encoder.col_names_dict = self.dataset_.tensor_frame.col_names_dict
        # self.module = self.module.__class__(col_stats=self.dataset_.col_stats,
        #                                     col_names_dict=self.dataset_.tensor_frame.col_names_dict,
        #                                     **self.module.__dict__
        return super().fit(self.dataset_.df, None, **fit_params)

    def predict(self, X: Dataset | DataFrame) -> NDArray[Any]:
        if isinstance(X, DataFrame):
            self.dataset_ = Dataset(
                X,
                {
                    k: v
                    for k, v in self.dataset_.col_to_stype.items()
                    if k not in (self.target_col, )
                },
                split_col=None,
                target_col=None,
                col_to_sep=self.col_to_sep,
                col_to_text_embedder_cfg=self.col_to_text_embedder_cfg,
                col_to_text_tokenizer_cfg=self.col_to_text_tokenizer_cfg,
                col_to_image_embedder_cfg=self.col_to_image_embedder_cfg,
                col_to_time_format=self.col_to_time_format,
            )
            # no need to materialize probably
        else:
            self.dataset_ = X
        return super().predict(self.dataset_.df)


# TODO: make this behave more like NeuralNetClassifier
class NeuralNetClassifierPytorchFrame(NeuralNetPytorchFrame):
    def fit(self, X: Dataset | DataFrame, y: ArrayLike | None = None,
            **fit_params):
        fit_result = super().fit(X, y, **fit_params)
        self.classes = getattr(
            self, "classes", None) or self.dataset_.df["target_col"].unique()
        return fit_result
