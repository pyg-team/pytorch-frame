import pytest

from torch_frame import TaskType, stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.gbdt import CatBoost


@pytest.mark.parametrize('task_type', [
    TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION
])
def test_catboost(task_type):
    dataset: Dataset = FakeDataset(num_rows=10, with_nan=True,
                                   stypes=[stype.numerical, stype.categorical],
                                   create_split=True, task_type=task_type)
    dataset.materialize()
    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')
    test_dataset = dataset.get_split_dataset('test')
    cb = CatBoost(
        task_type=task_type, num_classes=dataset.num_classes
        if task_type == TaskType.MULTICLASS_CLASSIFICATION else None)
    cb.tune(tf_train=train_dataset.tensor_frame,
            tf_val=val_dataset.tensor_frame, num_trials=2, num_boost_round=2)
    pred = cb.predict(tf_test=test_dataset.tensor_frame)
    # ensure the prediction is of shape [batch_size, 1]
    assert (pred.size(-1) == 1)
    score = cb.compute_metric(test_dataset.tensor_frame.y, pred)
    if task_type == TaskType.REGRESSION:
        # score is root mean squared error
        assert (score >= 0)
    else:
        # score is accuracy
        assert (0 <= score <= 1)
