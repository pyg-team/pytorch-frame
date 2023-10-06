import pytest

from torch_frame import TaskType, stype
from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.gbdt import CatBoost, XGBoost


@pytest.mark.parametrize('gbdt_cls', [
    CatBoost,
    XGBoost,
])
@pytest.mark.parametrize('task_type', [
    TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION
])
def test_catboost(gbdt_cls, task_type):
    dataset: Dataset = FakeDataset(num_rows=30, with_nan=True,
                                   stypes=[stype.numerical, stype.categorical],
                                   create_split=True, task_type=task_type)
    dataset.materialize()
    gbdt = gbdt_cls(
        task_type=task_type, num_classes=dataset.num_classes
        if task_type == TaskType.MULTICLASS_CLASSIFICATION else None)
    gbdt.tune(tf_train=dataset.tensor_frame, tf_val=dataset.tensor_frame,
              num_trials=2, num_boost_round=2)
    pred = gbdt.predict(tf_test=dataset.tensor_frame)
    score = gbdt.compute_metric(dataset.tensor_frame.y, pred)[gbdt.metric]
    if task_type == TaskType.REGRESSION:
        # score is root mean squared error
        assert (score >= 0)
    else:
        # score is accuracy
        assert (0 <= score <= 1)
