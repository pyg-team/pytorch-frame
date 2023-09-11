import torch
from category_encoders import CatBoostEncoder

from torch_frame import DataFrame, TensorFrame, stype
from torch_frame.transforms import FittableBaseTransform


class CategoricalCatBoostEncoder(FittableBaseTransform):
    r"""Encode the categorical features of :class:`TensorFrame` using
        CatBoostEncoder.

    """
    def _fit(self, tf_train: TensorFrame):
        if tf_train.y is None:
            raise RuntimeError(
                "'{self.__class__.__name__}' cannot be used when target column"
                " is None.")
        if stype.categorical not in tf_train.col_names_dict:
            print("The input TensorFrame does not contain any categorical "
                  "columns. No fitting will be performed.")
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
        self.reordered_col_names = tf_train.col_names_dict[
            stype.numerical] + tf_train.col_names_dict[stype.categorical]

    def _forward(self, tf: TensorFrame) -> TensorFrame:
        if stype.categorical not in tf.col_names_dict:
            print("The input TensorFrame does not contain any categorical "
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
        if len(tf.col_names_dict[stype.numerical]) == 0:
            tf.x_dict[stype.numerical] = transformed_tensor
        else:
            tf.x_dict[stype.numerical] = torch.cat(
                (tf.x_dict[stype.numerical], transformed_tensor), dim=1)
        tf.col_names_dict[stype.numerical] = self.reordered_col_names.copy()

        # delete the categorical features
        del tf.col_names_dict[stype.categorical]
        del tf.x_dict[stype.categorical]

        return tf
