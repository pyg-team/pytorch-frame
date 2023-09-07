import torch

from torch_frame import TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.transforms import CategoricalCatBoostEncoder


def test_categorical_catboost_encoder_on_categorical_features_only_dataset():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.categorical],
                                   create_split=True)
    dataset.materialize()
    total_cols = len(dataset.feat_cols)
    categorical_features = dataset.tensor_frame.col_names_dict[
        stype.categorical]
    # the dataset only contains categorical features
    assert (total_cols == len(categorical_features))

    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    transform = CategoricalCatBoostEncoder(train_dataset.tensor_frame)
    out = transform(tensor_frame)

    # assert that there are no categorical features
    assert (stype.categorical not in out.col_names_dict)
    assert (stype.categorical not in out.x_dict)

    # assert that all features are numerical
    assert (len(out.col_names_dict[stype.numerical]) == total_cols)

    # assert that all categorical features are turned into numerical features
    assert (out.col_names_dict[stype.numerical] == categorical_features)


def test_categorical_catboost_encoder():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.numerical, stype.categorical],
                                   create_split=True)
    dataset.materialize()
    total_cols = len(dataset.feat_cols)
    total_num_cols = len(dataset.tensor_frame.col_names_dict[stype.numerical])
    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    transform = CategoricalCatBoostEncoder()
    transform.fit(train_dataset.tensor_frame)
    out = transform(tensor_frame)

    # assert that there are no categorical features
    assert (stype.categorical not in out.col_names_dict)
    assert (stype.categorical not in out.x_dict)

    # assert that all features are numerical
    assert (len(out.col_names_dict[stype.numerical]) == total_cols)

    # assert that the numerical features are unchanged
    assert (torch.eq(dataset.tensor_frame.x_dict[stype.numerical],
                     out.x_dict[stype.numerical][:, :total_num_cols]).all())
    assert (dataset.tensor_frame.col_names_dict[stype.numerical] ==
            out.col_names_dict[stype.numerical][:total_num_cols])
