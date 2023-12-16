from __future__ import annotations

import pandas as pd

import torch_frame
from torch_frame import stype
from torch_frame.utils.split import SPLIT_TO_NUM


class HuggingFaceDatasetDict(torch_frame.data.Dataset):
    r"""Load a Hugging Face :obj:`datasets.DatasetDict` dataset
    to a :obj:`torch_frame.data.Dataset` with pre-defined split information.

    Args:
        path (str): Path or name of the dataset.

    Example:
        Load the `emotion` dataset from the Hugging Face Hub to the
        :obj:`torch_frame.data.Dataset`:

    .. code-block:: python

        >>> import torch_frame
        >>> from torch_frame.datasets import HuggingFaceDatasetDict
        >>> from torch_frame.config.text_embedder import TextEmbedderConfig
        >>> from torch_frame.testing.text_embedder import HashTextEmbedder
        >>> dataset = HuggingFaceDatasetDict(
        ...     path="emotion",
        ...     col_to_stype={
        ...         "text": torch_frame.text_embedded,
        ...         "label": torch_frame.categorical
        ...     },
        ...     target_col="label",
        ...     col_to_text_embedder_cfg=TextEmbedderConfig(
        ...         text_embedder=HashTextEmbedder(10)),
        ... )
        >>> dataset.materialize()
        >>> dataset.tensor_frame
        TensorFrame(
            num_cols=1,
            num_rows=20000,
            text_embedded (1): ['text'],
            has_target=True,
            device='cpu',
        )

    """
    def __init__(
        self,
        path: str,
        col_to_stype: dict[str, stype],
        target_col: str | None = None,
        **kwargs,
    ) -> None:
        try:
            from datasets import DatasetDict, load_dataset
        except ImportError:  # pragma: no cover
            raise ImportError("Please run `pip install datasets` at first.")
        dataset = load_dataset(path)
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"{self.__class__} only supports `DatasetDict`")
        # Convert dataset to pandas format
        dataset.set_format(type="pandas")
        dfs = []
        for split_name in dataset:
            # Load pandas dataframe for each split
            df: pd.DataFrame = dataset[split_name][:]

            # Transform HF dataset split to `SPLIT_TO_NUM` accepted one:
            if "train" in split_name:
                split_name = "train"
            elif "val" in split_name:
                # Some datasets have val split name as `"validation"`,
                # here we transform it to `"val"`:
                split_name = "val"
            elif "test" in split_name:
                split_name = "test"
            else:
                raise ValueError(f"Invalid split name: '{split_name}'. "
                                 f"Expected one of the following PyTorch "
                                 f"Frame Dataset split names: "
                                 f"{list(SPLIT_TO_NUM.keys())}.")

            # Add the split column
            df = df.assign(split=SPLIT_TO_NUM[split_name])
            dfs.append(df)
        df = pd.concat(dfs)
        super().__init__(df, col_to_stype, target_col=target_col,
                         split_col='split', **kwargs)
