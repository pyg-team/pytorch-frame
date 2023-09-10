from torch_frame import TaskType, stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.gbdt import XGBoost


def test_excelformer():
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=False,
                                   stypes=[stype.numerical, stype.categorical],
                                   create_split=True,
                                   task_type=TaskType.REGRESSION)
    dataset.materialize()
    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')
    test_dataset = dataset.get_split_dataset('test')
    XGB = XGBoost(task_type=TaskType.MULTICLASS_CLASSIFICATION)
    XGB.fit_tune(tf_train=train_dataset.tensor_frame,
                 tf_val=val_dataset.tensor_frame, num_trials=2,
                 num_boost_round=2)
    test_acc = XGB.eval(tf_test=test_dataset.tensor_frame)
    assert (test_acc >= 0 and test_acc <= 1)
