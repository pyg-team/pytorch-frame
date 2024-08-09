import os
from typing import Optional

import pandas as pd

import torch_frame
from torch_frame import stype
from torch_frame.utils.infer_stype import infer_series_stype


class OpenMLDataset(torch_frame.data.Dataset):
    r"""The `OpenML<https://www.openml.org/.`_ Data Collection.
    designed to integrate with the torch_frame library.

    More detailed information about the OpenML dataset can be found at
    `Datasets on OpenML
    <https://www.openml.org/search?type=data>`_.

    Args:
        dataset_id (int): The ID of the dataset to be loaded from OpenML.
        cache_dir (str, optional): The directory where  the dataset is cached.
                If None, the default cache directory is used.
    """
    def __init__(self, dataset_id: int, cache_dir: Optional[str] = None):
        try:      
            import openml
        except ImportError:
            raise ImportError(
                "The OpenML library is required to use the OpenMLDataset class. "
                "You can install it using `pip install openml`."
            )
        if cache_dir is not None:
            openml.config.set_root_cache_directory(
                os.path.expanduser(cache_dir))
        self.dataset_id = dataset_id
        self._openml_dataset = openml.datasets.get_dataset(
            self.dataset_id,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        # Get dataset info from OpenML
        self.dataset_info = self._openml_dataset.qualities
        target_col = self._openml_dataset.default_target_attribute
        X, y, self.categorical_indicator, _ = self._openml_dataset.get_data(
            target=target_col)
        df = pd.concat([X, y], axis=1)
        self._task_type: torch_frame.TaskType = (
            torch_frame.TaskType.BINARY_CLASSIFICATION)
        self._num_classes: int = 0

        # The column type can be inferred from the categorical_indicator
        col_to_stype = {
            col:
            stype.categorical
            if self.categorical_indicator[i] else stype.numerical
            for i, col in enumerate(X.columns)
        }

        # Infer the stype of the target column
        target_col_type = infer_series_stype(df[target_col])
        if target_col_type == torch_frame.categorical:
            assert self.dataset_info["NumberOfClasses"] > 0
            if self.dataset_info["NumberOfClasses"] == 2:
                assert df[target_col].nunique() == 2
                self._task_type = torch_frame.TaskType.BINARY_CLASSIFICATION
                self._num_classes = 2
            else:
                assert df[target_col].nunique(
                ) == self.dataset_info["NumberOfClasses"]
                self._task_type = (
                    torch_frame.TaskType.MULTICLASS_CLASSIFICATION)
                self._num_classes = int(self.dataset_info["NumberOfClasses"])
            col_to_stype[target_col] = torch_frame.categorical
        else:
            assert self.dataset_info["NumberOfClasses"] == 0
            self._task_type = torch_frame.TaskType.REGRESSION
            self._num_classes = 0
            col_to_stype[target_col] = torch_frame.numerical

        super().__init__(df=df, col_to_stype=col_to_stype,
                         target_col=target_col)

    # NOTE: Overriding the `task_type()` and `num_classes` property method
    @property
    def task_type(self) -> torch_frame.TaskType:
        """Returns the task type of the dataset.

        Returns:
            torch_frame.TaskType: The task type of the dataset.
        """
        return self._task_type

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset.

        Returns:
            int: The number of classes in the dataset.
        """
        return self._num_classes
