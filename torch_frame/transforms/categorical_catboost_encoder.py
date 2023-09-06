import torch
from category_encoders import CatBoostEncoder

from torch_frame import TensorFrame, stype
from torch_frame.transforms import BaseTransform


class CategoricalCatboostEncoder(BaseTransform):
    r"""Base class for transform that transforms the input tensorflow
    to output tensorflow."""
    def __init__(self, tf_train: TensorFrame):
        self.encoder = CatBoostEncoder(
            cols=tf_train.col_names_dict[stype.categorical]).fit(
                tf_train.x_dict[stype.categorical], tf_train.y)
        return

    def forward(self, tf: TensorFrame) -> TensorFrame:
        r"""Process column-wise 3-dimensional tensor into another column-wise
        3-dimensional tensor.

        Args:
            tf (TensorFrame): Input column-wise tensor of shape
                :Tensorframe:`[batch_size, num_cols, hidden_channels]`.
        """
        if len(tf.col_names_dict[stype.categorical] == 0):
            return tf

        tf.x_dict[stype.categorical] = self.encoder.transform(
            tf.x_dict[stype.categorical])

        tf.x_dict[stype.numerical] = torch.cat(
            (tf.x_dict[stype.numerical], tf.x_dict[stype.categorical]), dim=1)
        tf.col_names_dict[stype.numerical] = tf.col_names_dict[
            stype.numerical] + tf.col_names_dict[stype.categorical]
        tf.col_names_dict[stype.categorical] = []
        return tf
