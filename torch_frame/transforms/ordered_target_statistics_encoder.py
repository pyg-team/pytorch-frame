import logging
from typing import Any, Dict

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.transforms import FittableBaseTransform


class OrderedTargetStatisticsEncoder(FittableBaseTransform):
    r"""A transform that encodes the categorical features of
        :class:`TensorFrame` using ordered target statistics.
        The original encoding is explained in
        https://arxiv.org/abs/1706.09516
    """
    def _replace_nans(self, x: Tensor):
        r"""Replace NaNs with most frequent class.

        Args:
            tf (TensorFrame): Input :obj:`TensorFrame` whose NaN values
                in categorical columns are to be replaced.
        Returns:
            tf (Tensor): Output :obj:`TensorFrame` with NaN values replaced.
        """
        x = x.clone()
        for col in range(x.size(1)):
            column_data = x[:, col]
            nan_mask = column_data < 0
            if not nan_mask.any():
                continue
            fill_value = 0.
            column_data[nan_mask] = fill_value
        return x

    def _fit(self, tf_train: TensorFrame, col_stats: Dict[str, Dict[StatType,
                                                                    Any]]):
        if tf_train.y is None:
            raise RuntimeError(
                "'{self.__class__.__name__}' cannot be used when target column"
                " is None.")
        if stype.categorical not in tf_train.col_names_dict:
            logging.info(
                "The input TensorFrame does not contain any categorical "
                "columns. No fitting will be performed.")
            self._transformed_stats = col_stats
            return

        self.target_mean = torch.mean(tf_train.y)
        self.col_stats = col_stats

    def _forward(self, tf: TensorFrame) -> TensorFrame:
        if stype.categorical not in tf.col_names_dict:
            logging.info(
                "The input TensorFrame does not contain any categorical "
                "columns. The original TensorFrame will be returned.")
            return tf

        # TODO: When time stype is added, make sure the order is based on
        # time if time column exists.
        perm_idx = torch.randperm(len(tf.y))
        permuted_tensor = tf.feat_dict[stype.categorical][perm_idx]
        permuted_tensor = self._replace_nans(permuted_tensor)
        transformed_tensor = torch.zeros(tf.feat_dict[stype.categorical].shape)

        for i in range(len(tf.col_names_dict[stype.categorical])):
            col_name = tf.col_names_dict[stype.categorical][i]
            num_classes = len(self.col_stats[col_name][StatType.COUNT][0])
            feat = permuted_tensor[:, i]
            one_hot_col = F.one_hot(feat, num_classes)
            accum_sum = torch.cumsum(one_hot_col,
                                     dim=0)[torch.arange(len(tf.y)), feat]
            transformed_tensor[:, i] = (accum_sum * tf.y +
                                        self.target_mean) / (accum_sum + 1)

        transformed_df = pd.DataFrame(
            transformed_tensor.cpu().numpy(),
            columns=tf.col_names_dict[stype.categorical])

        transformed_col_stats = self.col_stats.copy()
        for col in tf.col_names_dict[stype.categorical]:
            # TODO: Make col stats computed purely with PyTorch
            # (without mapping back to pandas series).
            transformed_col_stats[col] = compute_col_stats(
                transformed_df[col], stype.numerical)

        self._transformed_stats = transformed_col_stats

        # turn the categorical features into numerical features
        if stype.numerical in tf.feat_dict:
            tf.feat_dict[stype.numerical] = torch.cat(
                (tf.feat_dict[stype.numerical], transformed_tensor), dim=1)
            tf.col_names_dict[stype.numerical] = tf.col_names_dict[
                stype.numerical] + tf.col_names_dict[stype.categorical]
        else:
            tf.feat_dict[stype.numerical] = transformed_tensor
            tf.col_names_dict[stype.numerical] = tf.col_names_dict[
                stype.categorical]

        # delete the categorical features
        tf.col_names_dict.pop(stype.categorical)
        tf.feat_dict.pop(stype.categorical)

        return tf
