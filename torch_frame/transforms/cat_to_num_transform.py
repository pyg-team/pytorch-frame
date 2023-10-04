import logging
from typing import Any, Dict

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_frame import TensorFrame, stype, NAStrategy
from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.transforms import FittableBaseTransform


class CatToNumTransform(FittableBaseTransform):
    r"""A transform that encodes the categorical features of
        :class:`TensorFrame` using ordered target statistics.
        The original encoding is explained in
        https://arxiv.org/abs/1706.09516
    """
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

        tensor = self._replace_nans(tf_train.feat_dict[stype.categorical], NAStrategy.MOST_FREQUENT)
        self.col_stats = col_stats
        columns = []
        # check if it is multi class classification task
        if tf_train.y.dtype == torch.long and tf_train.y.max() > 1:
            num_classes = tf_train.y.max() + 1
            target = F.one_hot(tf_train.y, num_classes)[:,:-1]
            self.target_mean = target.float().mean(dim=0)
            shape = tf_train.feat_dict[stype.categorical].shape
            transformed_tensor = torch.zeros(shape[0], shape[1] * (num_classes -1))
            for i in range(len(tf_train.col_names_dict[stype.categorical])):
                col_name = tf_train.col_names_dict[stype.categorical][i]
                count = torch.tensor(col_stats[col_name][StatType.COUNT][1])
                feat = tensor[:, i]
                v = torch.index_select(count, 0, feat).unsqueeze(1).repeat(1, num_classes-1)
                transformed_tensor[:, i * (num_classes - 1):(i + 1)* (num_classes - 1)] = (v * target + self.target_mean) / (v + 1)
                columns += [col_name + f"_{i}" for i in range(num_classes - 1)]
        else:
            target = tf_train.y
            self.target_mean = torch.mean(target.float())
            transformed_tensor = torch.zeros_like(tf_train.feat_dict[stype.categorical])
            for i in range(len(tf_train.col_names_dict[stype.categorical])):
                col_name = tf_train.col_names_dict[stype.categorical][i]
                count = torch.tensor(col_stats[col_name][StatType.COUNT][1])
                feat = tensor[:, i]
                v = torch.index_select(count, 0, feat)
                transformed_tensor[:, i] = (v * target + self.target_mean) / (v + 1)
                columns.append(col_name)

        transformed_df = pd.DataFrame(
            transformed_tensor.cpu().numpy(),
            columns=columns)

        transformed_col_stats = dict()
        for col in columns:
            # TODO: Make col stats computed purely with PyTorch
            # (without mapping back to pandas series).
            transformed_col_stats[col] = compute_col_stats(
                transformed_df[col], stype.numerical)

        self._transformed_stats = transformed_col_stats


    def _forward(self, tf: TensorFrame) -> TensorFrame:
        if stype.categorical not in tf.col_names_dict:
            logging.info(
                "The input TensorFrame does not contain any categorical "
                "columns. The original TensorFrame will be returned.")
            return tf
        tensor = self._replace_nans(tf.feat_dict[stype.categorical], NAStrategy.MOST_FREQUENT)
        columns = []
        if tf.y.dtype == torch.long and tf.y.max() > 1:
            num_classes = tf.y.max() + 1
            target = F.one_hot(tf.y, num_classes)[:,:-1]
            self.target_mean = target.float().mean(dim=0)
            shape = tf.feat_dict[stype.categorical].shape
            transformed_tensor = torch.zeros(shape[0], shape[1] * (num_classes -1))
            for i in range(len(tf.col_names_dict[stype.categorical])):
                col_name = tf.col_names_dict[stype.categorical][i]
                count = torch.tensor(self.col_stats[col_name][StatType.COUNT][1])
                feat = tensor[:, i]
                v = torch.index_select(count, 0, feat).unsqueeze(1).repeat(1, num_classes-1)
                transformed_tensor[:, i * (num_classes - 1):(i + 1)* (num_classes - 1)] = (v * target + self.target_mean) / (v + 1)
                columns += [col_name + f"_{i}" for i in range(num_classes - 1)]
        else:
            target = tf.y
            self.target_mean = torch.mean(target.float())
            transformed_tensor = torch.zeros_like(tf.feat_dict[stype.categorical])
            for i in range(len(tf.col_names_dict[stype.categorical])):
                col_name = tf.col_names_dict[stype.categorical][i]
                count = torch.tensor(self.col_stats[col_name][StatType.COUNT][1])
                feat = tensor[:, i]
                v = torch.index_select(count, 0, feat)
                transformed_tensor[:, i] = (v * target + self.target_mean) / (v + 1)
                columns.append(col_name)

        # turn the categorical features into numerical features
        if stype.numerical in tf.feat_dict:
            tf.feat_dict[stype.numerical] = torch.cat(
                (tf.feat_dict[stype.numerical], transformed_tensor), dim=1)
            tf.col_names_dict[stype.numerical] = tf.col_names_dict[
                stype.numerical] + columns
        else:
            tf.feat_dict[stype.numerical] = transformed_tensor
            tf.col_names_dict[stype.numerical] = columns
        # delete the categorical features
        tf.col_names_dict.pop(stype.categorical)
        tf.feat_dict.pop(stype.categorical)

        return tf
