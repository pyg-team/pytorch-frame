import os.path as osp

import pandas as pd
from PIL import Image

import torch_frame
from torch_frame.config import ImageEmbedderConfig


class HousePricesAndImages(torch_frame.data.Dataset):
    r"""The `House Prices and Images
    <https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal>`_
    dataset from Kaggle.
    """

    url = ''  # noqa

    def __init__(
        self,
        root: str,
        col_to_image_embedder_cfg: ImageEmbedderConfig
        | dict[str, ImageEmbedderConfig],
    ):
        # path = self.download_url(self.url, root)
        path = "/Users/zecheng/code/pytorch-frame/house_prices_and_images"
        csv_path = osp.join(path, "data.csv")
        image_folder = osp.join(path, "images")
        df = pd.read_csv(csv_path)
        images = []
        for image_id in df["image_id"]:
            image = Image.open(osp.join(image_folder, f"{image_id}.jpg"))
            images.append(image.copy())
            image.close()
        df = df.rename(columns={'citi': 'city'})
        df = df.drop(columns=['image_id', 'n_citi'])
        image_df = pd.DataFrame({'image': images})
        df = pd.concat([df, image_df], axis=1)

        col_to_stype = {
            "city": torch_frame.categorical,
            "bed": torch_frame.numerical,
            "bath": torch_frame.numerical,
            "sqft": torch_frame.numerical,
            "price": torch_frame.numerical,
            "image": torch_frame.image_embedded,
        }

        super().__init__(df, col_to_stype, target_col="price",
                         col_to_image_embedder_cfg=col_to_image_embedder_cfg)
