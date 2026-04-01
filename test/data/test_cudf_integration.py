"""Integration tests for cuDF DataFrame support.

All tests are skipped when cuDF is not installed.
These mirror key existing pandas tests but use cuDF DataFrames.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import torch_frame
from torch_frame._dataframe_compat import is_cudf_available
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType

cudf = pytest.importorskip("cudf")

pytestmark = pytest.mark.skipif(
    not is_cudf_available(),
    reason="cuDF not installed",
)


def _to_cudf(df: pd.DataFrame) -> "cudf.DataFrame":
    """Convert a pandas DataFrame to cuDF."""
    return cudf.DataFrame.from_pandas(df)


class TestCudfMaterialize:
    """Test that materialize() works with cuDF DataFrames and produces
    the same results as pandas."""

    def test_numerical(self):
        df = pd.DataFrame({
            "num_1": [1.0, 2.0, 3.0, 4.0],
            "num_2": [10.0, 20.0, 30.0, 40.0],
            "target": [0.1, 0.2, 0.3, 0.4],
        })
        col_to_stype = {
            "num_1": torch_frame.numerical,
            "num_2": torch_frame.numerical,
            "target": torch_frame.numerical,
        }

        ds_pd = Dataset(df, col_to_stype, target_col="target").materialize()
        ds_cu = Dataset(
            _to_cudf(df), col_to_stype, target_col="target"
        ).materialize()

        tf_pd = ds_pd.tensor_frame
        tf_cu = ds_cu.tensor_frame
        assert torch.allclose(
            tf_pd.feat_dict[torch_frame.numerical],
            tf_cu.feat_dict[torch_frame.numerical],
        )
        assert torch.equal(tf_pd.y, tf_cu.y)

    def test_categorical(self):
        df = pd.DataFrame({
            "cat_1": [0, 1, 2, 1, 0],
            "target": [0, 1, 0, 1, 0],
        })
        col_to_stype = {
            "cat_1": torch_frame.categorical,
            "target": torch_frame.categorical,
        }

        ds_pd = Dataset(df, col_to_stype, target_col="target").materialize()
        ds_cu = Dataset(
            _to_cudf(df), col_to_stype, target_col="target"
        ).materialize()

        assert torch.equal(
            ds_pd.tensor_frame.feat_dict[torch_frame.categorical],
            ds_cu.tensor_frame.feat_dict[torch_frame.categorical],
        )
        assert torch.equal(ds_pd.tensor_frame.y, ds_cu.tensor_frame.y)

    def test_multicategorical(self):
        data = {"multicat": ["A|B", "B|C|A", "", "B", "B|A|A", None]}
        df = pd.DataFrame(data)
        col_to_stype = {"multicat": torch_frame.multicategorical}

        ds_pd = Dataset(
            df, col_to_stype, col_to_sep={"multicat": "|"}
        ).materialize()
        ds_cu = Dataset(
            _to_cudf(df), col_to_stype, col_to_sep={"multicat": "|"}
        ).materialize()

        feat_pd = ds_pd.tensor_frame.feat_dict[torch_frame.multicategorical]
        feat_cu = ds_cu.tensor_frame.feat_dict[torch_frame.multicategorical]

        # Compare row by row (order within a row may differ)
        for i in range(len(df)):
            pd_vals = feat_pd[i, 0].sort().values
            cu_vals = feat_cu[i, 0].sort().values
            assert torch.equal(pd_vals, cu_vals), (
                f"Row {i}: pandas={pd_vals} vs cudf={cu_vals}")

        # Compare col_stats
        pd_stats = ds_pd.col_stats["multicat"][StatType.MULTI_COUNT]
        cu_stats = ds_cu.col_stats["multicat"][StatType.MULTI_COUNT]
        assert pd_stats[0] == cu_stats[0]
        assert pd_stats[1] == cu_stats[1]

    def test_timestamp(self):
        df = pd.DataFrame({
            "ts": [
                "2023-01-15 10:30:00",
                "2023-06-20 14:00:00",
                "2024-01-01 00:00:00",
            ],
            "target": [1.0, 2.0, 3.0],
        })
        col_to_stype = {
            "ts": torch_frame.timestamp,
            "target": torch_frame.numerical,
        }
        fmt = "%Y-%m-%d %H:%M:%S"

        ds_pd = Dataset(
            df, col_to_stype, target_col="target",
            col_to_time_format={"ts": fmt},
        ).materialize()
        ds_cu = Dataset(
            _to_cudf(df), col_to_stype, target_col="target",
            col_to_time_format={"ts": fmt},
        ).materialize()

        assert torch.equal(
            ds_pd.tensor_frame.feat_dict[torch_frame.numerical],
            ds_cu.tensor_frame.feat_dict[torch_frame.numerical],
        )

    def test_categorical_target_order(self):
        """Ensure binary classification label order is preserved."""
        df = pd.DataFrame({
            "cat_1": [0, 1, 1, 1],
            "cat_2": [0, 1, 1, 1],
        })
        col_to_stype = {
            "cat_1": torch_frame.categorical,
            "cat_2": torch_frame.categorical,
        }

        ds_pd = Dataset(
            df, col_to_stype, target_col="cat_2"
        ).materialize()
        ds_cu = Dataset(
            _to_cudf(df), col_to_stype, target_col="cat_2"
        ).materialize()

        assert torch.equal(ds_pd.tensor_frame.y, ds_cu.tensor_frame.y)
        assert torch.equal(
            ds_pd.tensor_frame.feat_dict[torch_frame.categorical],
            ds_cu.tensor_frame.feat_dict[torch_frame.categorical],
        )

    def test_index_select(self):
        df = pd.DataFrame({
            "num_1": np.random.randn(10),
            "cat_1": np.random.randint(0, 3, 10),
            "target": np.random.randn(10),
        })
        col_to_stype = {
            "num_1": torch_frame.numerical,
            "cat_1": torch_frame.categorical,
            "target": torch_frame.numerical,
        }

        ds = Dataset(
            _to_cudf(df), col_to_stype, target_col="target"
        ).materialize()
        assert len(ds) == 10
        assert len(ds[:5]) == 5
        assert len(ds[[1, 2, 3]]) == 3

    def test_numerical_with_nan(self):
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        df = pd.DataFrame({
            "num": arr,
            "target": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        col_to_stype = {
            "num": torch_frame.numerical,
            "target": torch_frame.numerical,
        }

        ds_pd = Dataset(df, col_to_stype, target_col="target").materialize()
        ds_cu = Dataset(
            _to_cudf(df), col_to_stype, target_col="target"
        ).materialize()

        pd_feat = ds_pd.tensor_frame.feat_dict[torch_frame.numerical]
        cu_feat = ds_cu.tensor_frame.feat_dict[torch_frame.numerical]
        # NaN == NaN is False, so check element-wise
        assert pd_feat.shape == cu_feat.shape
        nan_mask = torch.isnan(pd_feat)
        assert torch.equal(nan_mask, torch.isnan(cu_feat))
        assert torch.equal(pd_feat[~nan_mask], cu_feat[~nan_mask])
