import pytest

from torch_frame.datasets import DataFrameBenchmark
from torch_frame.typing import TaskType


@pytest.mark.parametrize('scale', ["small", "medium", "large"])
@pytest.mark.parametrize('task_type', [
    TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.REGRESSION
])
def test_dataframe_benchmark_match(task_type, scale):
    # Make sure task_type, scale, idx triple map to the fixed underlying
    # dataset. New dataset can be appeneded, but the existing mapping needes to
    # be preserved.
    datasets = DataFrameBenchmark.datasets_available(task_type=task_type,
                                                     scale=scale)
    if task_type == TaskType.BINARY_CLASSIFICATION:
        if scale == 'small':
            assert datasets[0] == ('AdultCensusIncome', {})
            assert datasets[1] == ('Mushroom', {})
            assert datasets[2] == ('BankMarketing', {})
            assert datasets[3] == ('TabularBenchmark', {
                'name': 'MagicTelescope'
            })
            assert datasets[4] == ('TabularBenchmark', {
                'name': 'bank-marketing'
            })
            assert datasets[5] == ('TabularBenchmark', {'name': 'california'})
            assert datasets[6] == ('TabularBenchmark', {'name': 'credit'})
            assert datasets[7] == ('TabularBenchmark', {
                'name': 'default-of-credit-card-clients'
            })
            assert datasets[8] == ('TabularBenchmark', {'name': 'electricity'})
            assert datasets[9] == ('TabularBenchmark', {
                'name': 'eye_movements'
            })
            assert datasets[10] == ('TabularBenchmark', {'name': 'heloc'})
            assert datasets[11] == ('TabularBenchmark', {'name': 'house_16H'})
            assert datasets[12] == ('TabularBenchmark', {'name': 'pol'})
            assert datasets[13] == ('Yandex', {'name': 'adult'})
        elif scale == 'medium':
            assert datasets[0] == ('Dota2', {})
            assert datasets[1] == ('KDDCensusIncome', {})
            assert datasets[2] == ('TabularBenchmark', {
                'name': 'Diabetes130US'
            })
            assert datasets[3] == ('TabularBenchmark', {'name': 'MiniBooNE'})
            assert datasets[4] == ('TabularBenchmark', {'name': 'albert'})
            assert datasets[5] == ('TabularBenchmark', {'name': 'covertype'})
            assert datasets[6] == ('TabularBenchmark', {'name': 'jannis'})
            assert datasets[7] == ('TabularBenchmark', {'name': 'road-safety'})
            assert datasets[8] == ('Yandex', {'name': 'higgs_small'})
        elif scale == 'large':
            assert datasets[0] == ('TabularBenchmark', {'name': 'Higgs'})
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if scale == 'small':
            assert len(datasets) == 0
        elif scale == 'medium':
            assert datasets[0] == ('Yandex', {'name': 'aloi'})
            assert datasets[1] == ('Yandex', {'name': 'helena'})
            assert datasets[2] == ('Yandex', {'name': 'jannis'})
        elif scale == 'large':
            assert datasets[0] == ('ForestCoverType', {})
            assert datasets[1] == ('PokerHand', {})
            assert datasets[2] == ('Yandex', {'name': 'covtype'})
    elif task_type == TaskType.REGRESSION:
        if scale == 'small':
            assert datasets[0] == ('TabularBenchmark', {
                'name': 'Bike_Sharing_Demand'
            })
            assert datasets[1] == ('TabularBenchmark', {
                'name': 'Brazilian_houses'
            })
            assert datasets[2] == ('TabularBenchmark', {'name': 'cpu_act'})
            assert datasets[3] == ('TabularBenchmark', {'name': 'elevators'})
            assert datasets[4] == ('TabularBenchmark', {'name': 'house_sales'})
            assert datasets[5] == ('TabularBenchmark', {'name': 'houses'})
            assert datasets[6] == ('TabularBenchmark', {'name': 'sulfur'})
            assert datasets[7] == ('TabularBenchmark', {
                'name': 'superconduct'
            })
            assert datasets[8] == ('TabularBenchmark', {'name': 'topo_2_1'})
            assert datasets[9] == ('TabularBenchmark', {
                'name': 'visualizing_soil'
            })
            assert datasets[10] == ('TabularBenchmark', {
                'name': 'wine_quality'
            })
            assert datasets[11] == ('TabularBenchmark', {'name': 'yprop_4_1'})
            assert datasets[12] == ('Yandex', {'name': 'california_housing'})
        elif scale == 'medium':
            assert datasets[0] == ('TabularBenchmark', {
                'name': 'Allstate_Claims_Severity'
            })
            assert datasets[1] == ('TabularBenchmark', {
                'name': 'SGEMM_GPU_kernel_performance'
            })
            assert datasets[2] == ('TabularBenchmark', {'name': 'diamonds'})
            assert datasets[3] == ('TabularBenchmark', {
                'name': 'medical_charges'
            })
            assert datasets[4] == ('TabularBenchmark', {
                'name': 'particulate-matter-ukair-2017'
            })
            assert datasets[5] == ('TabularBenchmark', {
                'name': 'seattlecrime6'
            })
        elif scale == 'large':
            assert datasets[0] == ('TabularBenchmark', {
                'name': 'Airlines_DepDelay_1M'
            })
            assert datasets[1] == ('TabularBenchmark', {
                'name': 'delays_zurich_transport'
            })
            assert datasets[2] == ('TabularBenchmark', {
                'name': 'nyc-taxi-green-dec-2016'
            })
            assert datasets[3] == ('Yandex', {'name': 'microsoft'})
            assert datasets[4] == ('Yandex', {'name': 'yahoo'})
            assert datasets[5] == ('Yandex', {'name': 'year'})


def test_dataframe_benchmark_object(tmp_path):
    dataset = DataFrameBenchmark(tmp_path, TaskType.BINARY_CLASSIFICATION,
                                 'small', 1)
    assert str(dataset) == ("DataFrameBenchmark(\n"
                            "  task_type=binary_classification,\n"
                            "  scale=small,\n"
                            "  idx=1,\n"
                            "  cls=Mushroom()\n"
                            ")")
    assert dataset.num_rows == 8124
    dataset.materialize()
