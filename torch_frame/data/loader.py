from __future__ import annotations

import torch

from torch_frame.data import Dataset, TensorFrame
from torch_frame.typing import IndexSelectType


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which creates mini-batches from a
    :class:`torch_frame.Dataset` or :class:`torch_frame.TensorFrame` object.

    .. code-block:: python

        import torch_frame

        dataset = ...

        loader = torch_frame.data.DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
        )

    Args:
        dataset (Dataset or TensorFrame): The dataset or tensor frame from
            which to load the data.
        *args (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
        **kwargs (optional): Additional keyword arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Dataset | TensorFrame,
        *args,
        **kwargs,
    ):
        kwargs.pop('collate_fn', None)

        if isinstance(dataset, Dataset):
            self.tensor_frame: TensorFrame = dataset.materialize().tensor_frame
        else:
            self.tensor_frame: TensorFrame = dataset

        super().__init__(
            range(len(dataset)),
            *args,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, index: IndexSelectType) -> TensorFrame:
        return self.tensor_frame[index]
