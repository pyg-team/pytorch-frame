import copy
import logging
from typing import Any, Dict

import pandas as pd
import torch
import torch.nn.functional as F

from torch_frame import NAStrategy, TensorFrame, stype
from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.transforms import FittableBaseTransform


class CatToNumTransform(FittableBaseTransform):
    r"""A transform that encodes the categorical features of
        :class:`TensorFrame` using target statistics.
        The original transform is explained in
        https://dl.acm.org/doi/10.1145/507533.507538
        Specifically, each categorical feature is transformed
        into numerical feature using m-probability estimate,
        defined by (n_c + p * m)/ (n + m), where n_c is the
        total count of the category, n is the total count,
        p is the prior probability and m is a smoothing factor.
    """
    def _fit(
        self,
        tf_train: TensorFrame,
        col_stats: Dict[str, Dict[StatType, Any]],
    ):
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
        tensor = self._replace_nans(tf_train.feat_dict[stype.categorical],
                                    NAStrategy.MOST_FREQUENT)
        self.col_stats = col_stats
        columns = []
        self.data_size = tensor.size(0)
        # Check if it is multiclass classification task.
        # If it is multiclass classification task, then it doesn't make sense
        # to assume the target mean as the prior. Therefore, we need to expand
        # the number of columns to (num_target_classes - 1). More details can
        # be found in https://dl.acm.org/doi/10.1145/507533.507538
        if not torch.is_floating_point(tf_train.y) and tf_train.y.max() > 1:
            num_classes = tf_train.y.max() + 1
            target = F.one_hot(tf_train.y, num_classes)[:, :-1]
            self.target_mean = target.float().mean(dim=0)
            shape = tf_train.feat_dict[stype.categorical].shape
            transformed_tensor = torch.zeros(shape[0],
                                             shape[1] * (num_classes - 1),
                                             dtype=torch.float32,
                                             device=tf_train.device)
        else:
            num_classes = 2
            target = tf_train.y.unsqueeze(1)
            self.target_mean = torch.mean(target.float())
            transformed_tensor = torch.zeros_like(
                tf_train.feat_dict[stype.categorical], dtype=torch.float32)

        for i in range(len(tf_train.col_names_dict[stype.categorical])):
            col_name = tf_train.col_names_dict[stype.categorical][i]
            count = torch.tensor(col_stats[col_name][StatType.COUNT][1],
                                 device=tf_train.device)
            feat = tensor[:, i]
            v = torch.index_select(count, 0, feat).unsqueeze(1).repeat(
                1, num_classes - 1)
            transformed_tensor[:, i * (num_classes - 1):(i + 1) *
                               (num_classes - 1)] = ((v + self.target_mean) /
                                                     (self.data_size + 1))
            columns += [col_name + f"_{i}" for i in range(num_classes - 1)]

        self.new_columns = columns
        transformed_df = pd.DataFrame(transformed_tensor.cpu().numpy(),
                                      columns=columns)

        transformed_col_stats = dict()
        if stype.numerical in tf_train.col_names_dict:
            for col in tf_train.col_names_dict[stype.numerical]:
                transformed_col_stats[col] = copy.copy(col_stats[col])
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
        tensor = self._replace_nans(tf.feat_dict[stype.categorical],
                                    NAStrategy.MOST_FREQUENT)
        if not torch.is_floating_point(tf.y) and tf.y.max() > 1:
            num_classes = tf.y.max() + 1
            shape = tf.feat_dict[stype.categorical].shape
            transformed_tensor = torch.zeros(shape[0],
                                             shape[1] * (num_classes - 1),
                                             dtype=torch.float32,
                                             device=tf.device)
        else:
            num_classes = 2
            transformed_tensor = torch.zeros_like(
                tf.feat_dict[stype.categorical], dtype=torch.float32)
        target_mean = self.target_mean.to(tf.device)
        for i in range(len(tf.col_names_dict[stype.categorical])):
            col_name = tf.col_names_dict[stype.categorical][i]
            count = torch.tensor(self.col_stats[col_name][StatType.COUNT][1],
                                 device=tf.device)
            feat = tensor[:, i]
            v = torch.index_select(count, 0, feat).unsqueeze(1).repeat(
                1, num_classes - 1)
            transformed_tensor[:, i * (num_classes - 1):(i + 1) *
                               (num_classes - 1)] = ((v + target_mean) /
                                                     (self.data_size + 1))

        # turn the categorical features into numerical features
        if stype.numerical in tf.feat_dict:
            tf.feat_dict[stype.numerical] = torch.cat(
                (tf.feat_dict[stype.numerical], transformed_tensor),
                dim=1).to(torch.float32)
            tf.col_names_dict[stype.numerical] = tf.col_names_dict[
                stype.numerical] + self.new_columns
        else:
            tf.feat_dict[stype.numerical] = transformed_tensor
            tf.col_names_dict[stype.numerical] = self.new_columns
        # delete the categorical features
        tf.col_names_dict.pop(stype.categorical)
        tf.feat_dict.pop(stype.categorical)

        return tf
