from torch_frame import TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.transforms import CategoricalCatboostEncoder


def test_categorical_catboost_encoder():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.categorical])
    dataset.materialize()
    total_cols = len(dataset.feat_cols)
    tensor_frame: TensorFrame = dataset.tensor_frame
    train_dataset = dataset.get_split_dataset('train')
    transform = CategoricalCatboostEncoder(train_dataset.tensor_frame)
    out = transform(tensor_frame)
    assert (len(out.col_names_dict[stype.categorical]) == 0)
    assert (len(out.col_names_dict[stype.numerical]) == total_cols)
