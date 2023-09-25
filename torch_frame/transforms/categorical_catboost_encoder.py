import logging
from typing import Any, Dict

import torch
from category_encoders import CatBoostEncoder

from torch_frame import DataFrame, TensorFrame, stype
from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.transforms import FittableBaseTransform


class CategoricalCatBoostEncoder(FittableBaseTransform):
    r"""Encode the categorical features of :class:`TensorFrame` using
        CatBoostEncoder.
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
            return
        # TODO: Implement the CatBoostEncoder with Pytorch rather than relying
        # on external library.
        self.encoder = CatBoostEncoder(
            cols=tf_train.col_names_dict[stype.categorical])
        # Converts the categorical columns of a :obj:`TensorFrame` into
        # :obj:`pd.DataFrame`. CatBoostEncoder does not take in numpy array or
        # tensor so we need to convert the TensorFrame obj to DataFrame first.
        df = DataFrame(data=tf_train.x_dict[stype.categorical].cpu(),
                       columns=tf_train.col_names_dict[stype.categorical])
        self.encoder.fit(df, tf_train.y.cpu())

        if stype.numerical in tf_train.col_names_dict:
            self.new_numerical_col_names = tf_train.col_names_dict[
                stype.numerical] + tf_train.col_names_dict[stype.categorical]
        else:
            self.new_numerical_col_names = tf_train.col_names_dict[
                stype.categorical]

        transformed_df = self.encoder.transform(df)
        transformed_col_stats = col_stats.copy()
        for col in tf_train.col_names_dict[stype.categorical]:
            # TODO: Make col stats computed purely with Pytorch
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
        # Converts the categorical columns of a :obj:`TensorFrame` into
        # :obj:`pd.DataFrame`. CatBoostEncoder does not take in numpy array or
        # tensor so we need to convert the TensorFrame obj to DataFrame first.
        df = DataFrame(data=tf.x_dict[stype.categorical],
                       columns=tf.col_names_dict[stype.categorical])
        transformed_tensor = torch.from_numpy(
            self.encoder.transform(df).values)

        # turn the categorical features into numerical features
        if stype.numerical in tf.x_dict:
            tf.x_dict[stype.numerical] = torch.cat(
                (tf.x_dict[stype.numerical], transformed_tensor), dim=1)
        else:
            tf.x_dict[stype.numerical] = transformed_tensor
        tf.col_names_dict[
            stype.numerical] = self.new_numerical_col_names.copy()

        # delete the categorical features
        tf.col_names_dict.pop(stype.categorical)
        tf.x_dict.pop(stype.categorical)

        return tf
