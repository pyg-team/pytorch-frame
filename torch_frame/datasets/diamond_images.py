from __future__ import annotations

import os.path as osp
import zipfile

import pandas as pd

import torch_frame
from torch_frame.config.image_embedder import ImageEmbedderConfig


class DiamondImages(torch_frame.data.Dataset):
    r"""The `Diamond Images
    <https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset>`_
    dataset from Kaggle.
    """

    url = ''  # noqa

    def __init__(
        self,
        root: str,
        col_to_image_embedder_cfg: ImageEmbedderConfig
        | dict[str, ImageEmbedderConfig],
        target: str = "colour",
    ):
        # path = self.download_url(self.url, root)
        path = "/Users/zecheng/code/pytorch-frame/diamond.zip"

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

        super().__init__(df, col_to_stype, target_col=target,
                         col_to_image_embedder_cfg=col_to_image_embedder_cfg)