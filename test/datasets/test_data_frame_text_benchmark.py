import tempfile

import pytest

from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.datasets import DataFrameTextBenchmark
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.typing import TaskType


@pytest.mark.parametrize('scale', ["small", "medium", "large"])
@pytest.mark.parametrize('task_type', [
    TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.REGRESSION
])
def test_data_frame_text_benchmark_match(task_type, scale):
    # Make sure task_type, scale, idx triple map to the fixed underlying
    # dataset. New dataset can be appended, but the existing mapping needs to
    # be preserved.
    datasets = DataFrameTextBenchmark.datasets_available(
        task_type=task_type, scale=scale)
    if task_type == TaskType.BINARY_CLASSIFICATION:
        if scale == 'small':
            assert datasets[0] == ('MultimodalTextBenchmark', {
                'name': 'fake_job_postings2'
            })
        elif scale == 'medium':
            assert datasets[0] == ('MultimodalTextBenchmark', {
                'name': 'jigsaw_unintended_bias100K'
            })
            assert datasets[1] == ('MultimodalTextBenchmark', {
                'name': 'kick_starter_funding'
            })
        elif scale == 'large':
            assert len(datasets) == 0
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if scale == 'small':
            assert datasets[0] == ('MultimodalTextBenchmark', {
                'name': 'product_sentiment_machine_hack'
            })
            assert datasets[1] == ('MultimodalTextBenchmark', {
                'name': 'news_channel'
            })
            assert datasets[2] == ('MultimodalTextBenchmark', {
                'name': 'data_scientist_salary'
            })
            assert datasets[3] == ('MultimodalTextBenchmark', {
                'name': 'melbourne_airbnb'
            })
        elif scale == 'medium':
            assert datasets[0] == ('MultimodalTextBenchmark', {
                'name': 'wine_reviews'
            })
            assert datasets[1] == ('HuggingFaceDatasetDict', {
                'path':
                'maharshipandya/spotify-tracks-dataset',
                'columns': [
                    'artists', 'album_name', 'track_name', 'popularity',
                    'duration_ms', 'explicit', 'danceability', 'energy', 'key',
                    'loudness', 'mode', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo',
                    'time_signature', 'track_genre'
                ],
                'target_col':
                'track_genre',
            })
        elif scale == 'large':
            assert datasets[0] == ('AmazonFineFoodReviews', {})
    elif task_type == TaskType.REGRESSION:
        if scale == 'small':
            assert datasets[0] == ('MultimodalTextBenchmark', {
                'name':
                'google_qa_answer_type_reason_explanation'
            })
            assert datasets[1] == ('MultimodalTextBenchmark', {
                'name':
                'google_qa_question_type_reason_explanation'
            })
            assert datasets[2] == ('MultimodalTextBenchmark', {
                'name': 'bookprice_prediction'
            })
            assert datasets[3] == ('MultimodalTextBenchmark', {
                'name': 'jc_penney_products'
            })
            assert datasets[4] == ('MultimodalTextBenchmark', {
                'name': 'women_clothing_review'
            })
            assert datasets[5] == ('MultimodalTextBenchmark', {
                'name': 'news_popularity2'
            })
            assert datasets[6] == ('MultimodalTextBenchmark', {
                'name': 'ae_price_prediction'
            })
            assert datasets[7] == ('MultimodalTextBenchmark', {
                'name': 'california_house_price'
            })
        elif scale == 'medium':
            assert datasets[0] == ('MultimodalTextBenchmark', {
                'name': 'mercari_price_suggestion100K'
            })
        elif scale == 'large':
            assert datasets[0] == ('Mercari', {})


def test_data_frame_text_benchmark_object():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = DataFrameTextBenchmark(
            root=temp_dir,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            scale='small',
            idx=0,
            col_to_text_embedder_cfg=TextEmbedderConfig(
                text_embedder=HashTextEmbedder(10)),
        )
    assert str(dataset) == ("DataFrameTextBenchmark(\n"
                            "  task_type=multiclass_classification,\n"
                            "  scale=small,\n"
                            "  idx=0,\n"
                            "  cls=MultimodalTextBenchmark()\n"
                            ")")
    assert dataset.num_rows == 6364
    dataset.materialize()
    train_dataset, val_dataset, test_dataset = dataset.split()
    assert train_dataset.num_rows + val_dataset.num_rows == 5091
    assert test_dataset.num_rows == 1273
