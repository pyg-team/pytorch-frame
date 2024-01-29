from typing import Any, List

import pandas as pd
import requests
import torch
from PIL import Image
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

import torch_frame
from torch_frame.config import ImageEmbedderConfig, TextEmbedderConfig
from torch_frame.data import Dataset


class CLIPDataset(Dataset):
    def __init__(
        self,
        url_to_images: List[str],
        sentences: List[str],
        col_to_text_embedder_cfg: TextEmbedderConfig,
        col_to_image_embedder_cfg: ImageEmbedderConfig,
    ):
        df = pd.DataFrame({
            "url_to_images": url_to_images,
            "sentences": sentences
        })
        col_to_stype = {
            "url_to_images": torch_frame.image_embedded,
            "sentences": torch_frame.text_embedded,
        }
        super().__init__(df, col_to_stype,
                         col_to_text_embedder_cfg=col_to_text_embedder_cfg,
                         col_to_image_embedder_cfg=col_to_image_embedder_cfg)


class TextToEmbedding:
    def __init__(self, clip_model: torch.nn.Module, processor: Any,
                 device: torch.device):
        self.processor = processor
        self.text_encoder = clip_model.text_model.to(device)
        self.text_projection = clip_model.text_projection.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, sentences: List[str]) -> Tensor:
        embeddings = []
        for sentence in sentences:
            inputs = self.processor(text=[sentence], return_tensors="pt",
                                    padding=True)
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }
            text_embeds = self.text_encoder(**inputs)[1]
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(
                p=2, dim=-1, keepdim=True)
            embeddings.append(text_embeds)
        return torch.cat(embeddings, dim=0)


class ImageToEmbedding:
    def __init__(self, clip_model: torch.nn.Module, processor: Any,
                 device: torch.device):
        self.processor = processor
        self.vision_encoder = clip_model.vision_model.to(device)
        self.visual_projection = clip_model.visual_projection.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, url_to_images: List[str]) -> Tensor:
        embeddings = []
        for url_to_image in url_to_images:
            image = Image.open(requests.get(url_to_image, stream=True).raw)
            inputs = self.processor(images=image, return_tensors="pt",
                                    padding=True)
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }
            image_embeds = self.vision_encoder(**inputs)[1]
            image_embeds = self.visual_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(
                p=2, dim=-1, keepdim=True).to("cpu")
            embeddings.append(image_embeds)
            image.close()
        return torch.cat(embeddings, dim=0)


class Decoder:
    def __init__(self, clip_model: torch.nn.Module):
        self.logit_scale = clip_model.logit_scale.exp()

    @torch.no_grad()
    def __call__(self, text_embeds: Tensor, image_embeds: Tensor) -> Tensor:
        logits_per_text = torch.matmul(text_embeds,
                                       image_embeds.t()) * self.logit_scale
        logits_per_image = logits_per_text.t()
        return logits_per_image


# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image_url = "https://hips.hearstapps.com/hmg-prod/images/little-cute-maltipoo-puppy-royalty-free-image-1652926025.jpg"
# sentences = ["a photo of a cat", "a photo of a dog"]

# image_url = "https://upload.wikimedia.org/wikipedia/commons/3/35/Hosico_cat_151788152_924821268345498_670082736362908526_n.jpg"
# sentences = [
#     "a photo of a yellow cat",
#     "a photo of a black cat",
#     "a photo of a white cat",
# ]

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/2020_Mercedes-Benz_GLC_300_4MATIC_in_Polar_White%2C_front_left.jpg/200px-2020_Mercedes-Benz_GLC_300_4MATIC_in_Polar_White%2C_front_left.jpg"
sentences = [
    "a photo of a black sedan",
    "a photo of a white sedan",
    "a photo of a black suv",
    "a photo of a white suv",
]

url_to_images = [image_url for _ in range(len(sentences))]

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_model.eval()
processor = CLIPProcessor.from_pretrained(model_name)

col_to_text_embedder_cfg = TextEmbedderConfig(
    text_embedder=TextToEmbedding(
        clip_model,
        processor,
        torch.device("cpu"),
    ),
    batch_size=1,
)
col_to_image_embedder_cfg = ImageEmbedderConfig(
    image_embedder=ImageToEmbedding(
        clip_model,
        processor,
        torch.device("cpu"),
    ),
    batch_size=1,
)
dataset = CLIPDataset(
    url_to_images=url_to_images,
    sentences=sentences,
    col_to_text_embedder_cfg=col_to_text_embedder_cfg,
    col_to_image_embedder_cfg=col_to_image_embedder_cfg,
)
dataset.materialize()
tensor_frame = dataset.tensor_frame

decoder = Decoder(clip_model)
feat = tensor_frame.feat_dict[torch_frame.embedding]
logits = []
for i in range(feat.size(0)):
    x = feat[i, :]
    text_embeds = x.values[:, x.offset[0]:x.offset[1]]
    image_embeds = x.values[:, x.offset[1]:x.offset[2]]
    logits.append(decoder(text_embeds, image_embeds))

probs = torch.softmax(torch.cat(logits, dim=0).view(-1), dim=0)
pred = sentences[int(probs.argmax(dim=0))]
print(f"Prediction for image {url_to_images[0]} is {pred} with probs {probs}")
