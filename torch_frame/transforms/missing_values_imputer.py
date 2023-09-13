from typing import List, Optional

import torch
from torch import Tensor

from torch_frame import ImputingStrategy, TensorFrame, stype
from torch_frame.transforms import FittableBaseTransform


class MissingValuesImputer(FittableBaseTransform):
    r"""Impute the missing values given strategy for each type of
        columns.

    Args:
        categorical_strategy (ImputingStrategy): Strategy to replace
            the NaN categorical values, can be either MOST_FREQUENT
            or ZEROS.
            (default ImputingStrategy.MOST_FREQUENT)
        numerical_strategy (ImputingStrategy): Strategy to replace
            the NaN numerical values, can be any of MEAN, MEDIAN
            or ZEROS. (default ImputingStrategy.MEAN)
    """
    def __init__(
        self,
        categorical_strategy: Optional[ImputingStrategy] = ImputingStrategy.
        MOST_FREQUENT,
        numerical_strategy: Optional[ImputingStrategy] = ImputingStrategy.MEAN
    ):
        super().__init__()
        if (categorical_strategy is not None
                and not categorical_strategy.is_categorical_strategy):
            raise RuntimeError(
                f"Cannot use {categorical_strategy} for categorical features.")
        if (numerical_strategy is not None
                and not numerical_strategy.is_numerical_strategy):
            raise RuntimeError(
                f"Cannot use {numerical_strategy} for numerical features.")
        self.strategy = dict()
        self.strategy[stype.numerical] = numerical_strategy
        self.strategy[stype.categorical] = categorical_strategy

    def _fit(self, tf_train: TensorFrame):
        self.fill_values = dict()
        for col_type in tf_train.col_names_dict:
            if self.strategy[col_type] is None:
                continue
            x = tf_train.x_dict[col_type]
            self.fill_values[col_type] = []
            strategy = self.strategy[col_type]
            for col in range(x.size(1)):
                column_data = x[:, col]
                if col_type == stype.numerical:
                    nan_mask = torch.isnan(column_data)
                else:
                    nan_mask = (column_data == -1)
                num_nans = nan_mask.sum().item()
                if num_nans == x.size(0):
                    raise ValueError(
                        f"Stats {strategy} cannot be computed"
                        "because column "
                        f"{tf_train.col_names_dict[col_type][col]}'s"
                        " data is invalid.")
                elif num_nans == 0:
                    continue
                valid_data = column_data[~nan_mask]
                if strategy == ImputingStrategy.MOST_FREQUENT:
                    unique_values, counts = valid_data.unique(
                        return_counts=True)
                    fill_value = unique_values[counts.argmax()]
                elif strategy == ImputingStrategy.MEAN:
                    fill_value = valid_data.mean()
                elif strategy == ImputingStrategy.MEDIAN:
                    fill_value = valid_data.median()
                elif strategy == ImputingStrategy.ZEROS:
                    fill_value = torch.tensor(
                        0) if col_type == stype.categorical else torch.tensor(
                            0.)
                self.fill_values[col_type].append(fill_value)

    def _replace_nans_with_fills(self, x: Tensor, fill_values: List[Tensor],
                                 col_type: stype):
        r"""Replace NaNs with fill values.

        Args:
            x (Tensor): Input :obj:`Tensor` whose NaN values are to be
                replaced.
            fill_values (List[Tensor]): A list of scalar Tensors containing
                the fill value for each column
            col_type (stype): Column type. If col_type is stype.categorical,
                the NaN values are -1's. Otherwise, the NaN values are NaNs.
        Returns:
            x (Tensor): Output :obj:`Tensor` with NaN values replaced.
        """
        for col in range(x.size(1)):
            column_data = x[:, col]
            if col_type == stype.numerical:
                nan_mask = torch.isnan(column_data)
            else:
                nan_mask = (column_data == -1)
            num_nans = nan_mask.sum().item()
            if num_nans == 0:
                continue
            column_data[nan_mask] = fill_values[col]
            x[:, col] = column_data
        return x

    def _forward(self, tf: TensorFrame) -> TensorFrame:
        for col_type in [stype.numerical, stype.categorical]:
            if col_type not in tf.col_names_dict or self.strategy[
                    col_type] is None:
                continue
            tf.x_dict[col_type] = self._replace_nans_with_fills(
                tf.x_dict[col_type], self.fill_values[col_type], col_type)
        return tf
