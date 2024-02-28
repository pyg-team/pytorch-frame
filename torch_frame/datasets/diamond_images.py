from __future__ import annotations

import os.path as osp
import zipfile

import pandas as pd

import torch_frame
from torch_frame.config.image_embedder import ImageEmbedderConfig


class DiamondImages(torch_frame.data.Dataset):
    r"""The `Diamond Images
    <https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset>`_
    dataset from Kaggle. The target is to predict :obj:`colour` of each
    diamond.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10 20 10
        :header-rows: 1

        * - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #cols (image)
          - #classes
          - Task
          - Missing value ratio
        * - 48,764
          - 4
          - 7
          - 1
          - 23
          - multiclass_classification
          - 0.167%
    """

    url = 'https://data.pyg.org/datasets/tables/diamond.zip'

    def __init__(
        self,
        root: str,
        col_to_image_embedder_cfg: ImageEmbedderConfig
        | dict[str, ImageEmbedderConfig],
    ):
        path = self.download_url(self.url, root)

        folder_path = osp.dirname(path)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(folder_path)

        subfolder_path = osp.join(folder_path, "diamond")
        csv_path = osp.join(subfolder_path, "diamond_data.csv")
        df = pd.read_csv(csv_path)
        df = df.drop(columns=["stock_number"])

        image_paths = []
        for path_to_img in df["path_to_img"]:
            path_to_img = path_to_img.replace("web_scraped/", "")
            image_paths.append(osp.join(subfolder_path, path_to_img))
        image_df = pd.DataFrame({"image_path": image_paths})
        df = pd.concat([df, image_df], axis=1)
        df = df.drop(columns=["path_to_img"])

        col_to_stype = {
            "shape": torch_frame.categorical,
            "carat": torch_frame.numerical,
            "clarity": torch_frame.categorical,
            "colour": torch_frame.categorical,
            "cut": torch_frame.categorical,
            "polish": torch_frame.categorical,
            "symmetry": torch_frame.categorical,
            "fluorescence": torch_frame.categorical,
            "lab": torch_frame.categorical,
            "length": torch_frame.numerical,
            "width": torch_frame.numerical,
            "depth": torch_frame.numerical,
            "image_path": torch_frame.image_embedded,
        }

        super().__init__(df, col_to_stype, target_col="colour",
                         col_to_image_embedder_cfg=col_to_image_embedder_cfg)
