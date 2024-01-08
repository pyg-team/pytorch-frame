from __future__ import annotations

import pandas as pd

import torch_frame
from torch_frame import stype
from torch_frame.utils.infer_stype import infer_df_stype
from torch_frame.utils.split import SPLIT_TO_NUM


class HuggingFaceDatasetDict(torch_frame.data.Dataset):
    r"""Load a Hugging Face :obj:`datasets.DatasetDict` dataset
    to a :obj:`torch_frame.data.Dataset` with pre-defined split information.
    To use this class, please install the `Datasets
    <https://huggingface.co/docs/datasets/installation>`_ package at first.
    For all available dataset paths and names, please refer to the
    `Hugging Face Datasets <https://huggingface.co/datasets>`_.

    Args:
        path (str): Path or name of the dataset.
        name (str, optional): Defining the name of the dataset configuration.
            (default: :obj:`None`)
        columns (list, optional): List of columns to be included.
            (default: :obj:`None`)

    Example:
        Load the `spotify-tracks-dataset` dataset from the Hugging Face Hub
        to the :obj:`torch_frame.data.Dataset`:

    .. code-block:: python

        >>> from torch_frame.datasets import HuggingFaceDatasetDict
        >>> from torch_frame.config.text_embedder import TextEmbedderConfig
        >>> from torch_frame.testing.text_embedder import HashTextEmbedder
        >>> dataset = HuggingFaceDatasetDict(
        ...     path="maharshipandya/spotify-tracks-dataset",
        ...     columns=["artists", "album_name", "track_name",
        ...              "popularity", "duration_ms", "explicit",
        ...              "danceability", "energy", "key", "loudness",
        ...              "mode", "speechiness", "acousticness",
        ...              "instrumentalness", "liveness", "valence",
        ...              "tempo", "time_signature", "track_genre"
        ...     ],
        ...     target_col="track_genre",
        ...     col_to_text_embedder_cfg=TextEmbedderConfig(
        ...         text_embedder=HashTextEmbedder(10)),
        ... )
        >>> dataset.materialize()
        >>> dataset.tensor_frame
        TensorFrame(
            num_cols=18,
            num_rows=114000,
            numerical (11): [
                'acousticness',
                'danceability',
                'duration_ms',
                'energy',
                'instrumentalness',
                'liveness',
                'loudness',
                'popularity',
                'speechiness',
                'tempo',
                'valence',
            ],
            categorical (4): [
                'explicit',
                'key',
                'mode',
                'time_signature',
            ],
            embedding (3): ['artists', 'album_name', 'track_name'],
            has_target=True,
            device='cpu',
        )

    """
    def __init__(
        self,
        path: str,
        name: str | None = None,
        columns: list[str] | None = None,
        col_to_stype: dict[str, stype] | None = None,
        target_col: str | None = None,
        **kwargs,
    ) -> None:
        try:
            from datasets import DatasetDict, load_dataset
        except ImportError:  # pragma: no cover
            raise ImportError("Please run `pip install datasets` at first.")
        dataset = load_dataset(path, name=name)
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"{self.__class__} only supports `DatasetDict`")
        # Convert dataset to pandas format
        dataset.set_format(type="pandas")
        dfs = []
        split_names = []

        for split_name in dataset:
            # Load pandas dataframe for each split
            df: pd.DataFrame = dataset[split_name][:]

            if columns is not None:
                df = df[columns]

            # Transform HF dataset split to `SPLIT_TO_NUM` accepted one:
            if "train" in split_name:
                split_names.append("train")
            elif "val" in split_name:
                # Some datasets have val split name as `"validation"`,
                # here we transform it to `"val"`:
                split_names.append("val")
            elif "test" in split_name:
                split_names.append("test")
            else:
                raise ValueError(f"Invalid split name: '{split_name}'. "
                                 f"Expected one of the following PyTorch "
                                 f"Frame Dataset split names: "
                                 f"{list(SPLIT_TO_NUM.keys())}.")
            dfs.append(df)

        # Only specify split if there are multiple splits:
        if len(split_names) > 1:
            dfs = [
                df.assign(split=SPLIT_TO_NUM[split_name])
                for split_name, df in zip(split_names, dfs)
            ]

        df = pd.concat(dfs).reset_index(drop=True)

        if col_to_stype is None:
            col_to_stype = infer_df_stype(df)

        if len(split_names) > 1:
            super().__init__(df, col_to_stype, target_col=target_col,
                             split_col='split', **kwargs)
        else:
            super().__init__(df, col_to_stype, target_col=target_col, **kwargs)
