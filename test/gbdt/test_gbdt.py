import os.path as osp
import tempfile

import pytest
import torch

from torch_frame import Metric, TaskType, stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.dataset import Dataset
from torch_frame.datasets.fake import FakeDataset
from torch_frame.gbdt import CatBoost, LightGBM, XGBoost
from torch_frame.testing.text_embedder import HashTextEmbedder


@pytest.mark.parametrize('gbdt_cls', [
    CatBoost,
    XGBoost,
    LightGBM,
])
@pytest.mark.parametrize('stypes', [
    [stype.numerical],
    [stype.categorical],
    [stype.text_embedded],
    [stype.numerical, stype.numerical, stype.text_embedded],
])
@pytest.mark.parametrize('task_type_and_metric', [
    (TaskType.REGRESSION, Metric.RMSE),
    (TaskType.REGRESSION, Metric.MAE),
    (TaskType.BINARY_CLASSIFICATION, Metric.ACCURACY),
    (TaskType.BINARY_CLASSIFICATION, Metric.ROCAUC),
    (TaskType.MULTICLASS_CLASSIFICATION, Metric.ACCURACY),
])
def test_gbdt_with_save_load(gbdt_cls, stypes, task_type_and_metric):
    task_type, metric = task_type_and_metric
    dataset: Dataset = FakeDataset(
        num_rows=30,
        with_nan=True,
        stypes=stypes,
        create_split=True,
        task_type=task_type,
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(8)),
    )
    dataset.materialize()
    gbdt = gbdt_cls(
        task_type=task_type,
        num_classes=dataset.num_classes
        if task_type == TaskType.MULTICLASS_CLASSIFICATION else None,
        metric=metric,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        path = osp.join(temp_dir, 'model.json')
        with pytest.raises(RuntimeError, match="is not yet fitted"):
            gbdt.save(path)

        if isinstance(gbdt_cls, XGBoost):
            gbdt.tune(tf_train=dataset.tensor_frame,
                      tf_val=dataset.tensor_frame, num_trials=2,
                      num_boost_round=1000, early_stopping_rounds=2)
            assert gbdt.model.best_iteration is not None
        else:
            gbdt.tune(
                tf_train=dataset.tensor_frame,
                tf_val=dataset.tensor_frame,
                num_trials=2,
                num_boost_round=2,
            )
        gbdt.save(path)

        loaded_gbdt = gbdt_cls(
            task_type=task_type,
            num_classes=dataset.num_classes
            if task_type == TaskType.MULTICLASS_CLASSIFICATION else None,
            metric=metric,
        )
        loaded_gbdt.load(path)

    pred = gbdt.predict(tf_test=dataset.tensor_frame)
    score = gbdt.compute_metric(dataset.tensor_frame.y, pred)

    loaded_score = loaded_gbdt.compute_metric(dataset.tensor_frame.y, pred)
    dataset.tensor_frame.y = None
    loaded_pred = loaded_gbdt.predict(tf_test=dataset.tensor_frame)

    assert torch.allclose(pred, loaded_pred, atol=1e-5)
    assert gbdt.metric == metric
    assert score == loaded_score
    if task_type == TaskType.REGRESSION:
        assert (score >= 0)
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        assert (0 <= score <= 1)
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        assert (0 <= score <= 1)
