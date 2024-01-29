from typing import Any, List

import pandas as pd
import requests
import torch
from PIL import Image
from torch import Tensor
from transformers import AutoProcessor, BlipForConditionalGeneration

import torch_frame
from torch_frame.config import ImageEmbedderConfig, TextTokenizerConfig
from torch_frame.data import Dataset
from torch_frame.typing import TextTokenizationOutputs


class BLIPDataset(Dataset):
    def __init__(
        self,
        url_to_images: List[str],
        sentences: List[str],
        col_to_text_tokenizer_cfg: TextTokenizerConfig,
        col_to_image_embedder_cfg: ImageEmbedderConfig,
    ):
        df = pd.DataFrame({
            "url_to_images": url_to_images,
            "sentences": sentences
        })
        col_to_stype = {
            "url_to_images": torch_frame.image_embedded,
            "sentences": torch_frame.text_tokenized,
        }
        super().__init__(df, col_to_stype,
                         col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
                         col_to_image_embedder_cfg=col_to_image_embedder_cfg)


class TextToToken:
    def __init__(self, processor: Any):
        self.processor = processor

    def tokenize(self, sentences: List[str]) -> TextTokenizationOutputs:
        return self.processor(text=sentences, return_tensors='pt')


class ImageToEmbedding:
    def __init__(self, blip_model: torch.nn.Module, processor: Any,
                 device: torch.device):
        self.processor = processor
        self.vision_encoder = blip_model.vision_model.to(device)
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
            image_embeds = self.vision_encoder(**inputs)[0]
            embeddings.append(image_embeds)
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings.squeeze(dim=0)


class BLIPDecoder(torch.nn.Module):
    def __init__(self, blip_model: torch.nn.Module, processor: Any):
        super().__init__()
        self.text_decoder = blip_model.text_decoder
        self.processor = processor

    @torch.no_grad()
    def forward(self, tf: torch_frame.TensorFrame) -> str:
        image_embeds = tf.feat_dict[torch_frame.embedding].values.unsqueeze(
            dim=0)
        text_inputs = {}
        for key in tf.feat_dict[torch_frame.text_tokenized]:
            text_inputs[key] = tf.feat_dict[
                torch_frame.text_tokenized][key].to_dense(
                    fill_value=0).squeeze(0)
        outputs = self.text_decoder.generate(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            encoder_hidden_states=image_embeds,
        )
        return self.processor.decode(outputs[0], skip_special_tokens=True)


model_name = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)
col_to_text_tokenizer_cfg = TextTokenizerConfig(
    text_tokenizer=TextToToken(processor).tokenize,
    batch_size=1,
)
col_to_image_embedder_cfg = ImageEmbedderConfig(
    image_embedder=ImageToEmbedding(
        model,
        processor,
        torch.device("cpu"),
    ),
    batch_size=1,
)
image_url = "https://hips.hearstapps.com/hmg-prod/images/little-cute-maltipoo-puppy-royalty-free-image-1652926025.jpg"
sentences = ["a photo of"]
url_to_images = [image_url for _ in range(len(sentences))]
dataset = BLIPDataset(
    url_to_images=url_to_images,
    sentences=sentences,
    col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
    col_to_image_embedder_cfg=col_to_image_embedder_cfg,
)
dataset.materialize()
decoder = BLIPDecoder(model, processor)
decoder(dataset.tensor_frame)
