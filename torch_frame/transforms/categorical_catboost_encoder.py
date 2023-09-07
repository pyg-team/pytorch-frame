import pandas as pd
import torch
from category_encoders import CatBoostEncoder

from torch_frame import DataFrame, TensorFrame, stype
from torch_frame.transforms import BaseTransform


class CategoricalCatboostEncoder(BaseTransform):
    r"""Encode the categorical features of :obj:`TensorFrame` using
        CatBoostEncoder.

        Args:
            tf_train (TensorFrame): TensorFrame containing the training data.
    """
    def __init__(self, tf_train: TensorFrame):
        if tf_train.y is None:
            raise RuntimeError(
                "CatBoostEncoder cannot be used when target column"
                " is None.")
        self.encoder = CatBoostEncoder(
            cols=tf_train.col_names_dict[stype.categorical])
        # Converts the categorical columns of a :obj:`TensorFrame` into
        # :obj:`pd.DataFrame`. CatBoostEncoder does not take in numpy array or
        # tensor so we need to convert the TensorFrame obj to DataFrame first.
        df = pd.DataFrame(data=tf_train.x_dict[stype.categorical],
                          columns=tf_train.col_names_dict[stype.categorical])
        self.encoder.fit(df, tf_train.y)

    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process TensorFrame obj into another TensorFrame obj.

        Args:
            tf (TensorFrame): Input :obj:`TensorFrame`.

        Returns:
            tf (TensorFrame): Input :obj:`TensorFrame` but with all
                the categorical columns converted to numerical
                columns using CatBoostEncoder.
        """
        if stype.categorical not in tf.col_names_dict:
            return tf
        # Converts the categorical columns of a :obj:`TensorFrame` into
        # :obj:`pd.DataFrame`. CatBoostEncoder does not take in numpy array or
        # tensor so we need to convert the TensorFrame obj to DataFrame first.
        df = DataFrame(data=tf.x_dict[stype.categorical],
                       columns=tf.col_names_dict[stype.categorical])
        transformed_tensor = torch.tensor(self.encoder.transform(df).values)

        # turn the categorical features into numerical features
        if len(tf.col_names_dict[stype.numerical]) == 0:
            tf.x_dict[stype.numerical] = transformed_tensor
        else:
            tf.x_dict[stype.numerical] = torch.cat(
                (tf.x_dict[stype.numerical], transformed_tensor), dim=1)
        tf.col_names_dict[stype.numerical] = tf.col_names_dict[
            stype.numerical] + tf.col_names_dict[stype.categorical]

        # delete the categorical features
        del tf.col_names_dict[stype.categorical]
        del tf.x_dict[stype.categorical]

        return tf
