import pandas as pd
import torch
from category_encoders import CatBoostEncoder

from torch_frame import TensorFrame, stype
from torch_frame.transforms import BaseTransform


class CategoricalCatboostEncoder(BaseTransform):
    r"""Class that use CatBoostEncoder to encode at the categorical features
        of a TensorFrame."""
    def __init__(self, tf_train: TensorFrame):
        r"""Fits the CatBoostEncoder with training data.

        Args:
            tf_train (TensorFrame): TensorFrame containing the training data.
                :Tensorframe:`[train_size, num_cols]`.
        """
        self.encoder = CatBoostEncoder(
            cols=tf_train.col_names_dict[stype.categorical])
        df = self._categorical_tf_to_df(tf_train)
        self.encoder.fit(df, tf_train.y)
        return

    def _categorical_tf_to_df(self, tf: TensorFrame) -> pd.DataFrame:
        r"""Converts the categorical columns of a TensorFrame obj into a
        pandas DataFrame. CatBoostEncoder does not take in numpy array or
        tensor so we need to convert the TensorFrame obj to DataFrame first.

        Args:
            tf (TensorFrame):  TensorFrame obj to be converted to pandas
                DataFrame object.
                :Tensorframe:`[train_size, num_cols]`.

        Returns:
            df (pd.DataFrame): Pandas DataFrame containing only the categorical
            columns of the TensorFrame obj.
        """
        df = pd.DataFrame(data=tf.x_dict[stype.categorical],
                          columns=tf.col_names_dict[stype.categorical])
        return df

    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process TensorFrame obj into another TensorFrame obj.

        Args:
            tf (TensorFrame): Input TensorFrame obj

        Returns:
            tf (TensorFrame): Input TensorFrame obj but with all
                the categorical columns converted to numerical
                columns using CatBoostEncoder.
        """
        if stype.categorical not in tf.col_names_dict:
            return tf
        df = self._categorical_tf_to_df(tf)
        tf.x_dict[stype.categorical] = torch.tensor(
            self.encoder.transform(df).values)

        # turn the categorical features into numerical features
        if len(tf.col_names_dict[stype.numerical]) == 0:
            tf.x_dict[stype.numerical] = tf.x_dict[stype.categorical]
        else:
            tf.x_dict[stype.numerical] = torch.cat(
                (tf.x_dict[stype.numerical], tf.x_dict[stype.categorical]),
                dim=1)
        tf.col_names_dict[stype.numerical] = tf.col_names_dict[
            stype.numerical] + tf.col_names_dict[stype.categorical]

        # delete the categorical features
        del tf.col_names_dict[stype.categorical]
        del tf.x_dict[stype.categorical]

        return tf
