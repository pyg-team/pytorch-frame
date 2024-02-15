import argparse
import os
import os.path as osp
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from torch_frame import stype
from torch_frame.config import ImageEmbedderConfig
from torch_frame.data import DataLoader
from torch_frame.datasets import DiamondImages
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
)

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--model",
    type=str,
    default="google/vit-base-patch16-224-in21k",
    choices=[
        "resnet18",
        "google/vit-base-patch16-224-in21k",
    ],
)

args = parser.parse_args()

# ResNet-18 - Image Embedded
# Best Val Acc: 0.9990, Best Test Acc: 0.9979
# ViT - Image Embedded
# Best Val Acc: 0.9990, Best Test Acc: 0.9979


class ImageToEmbedding:
    def __init__(self, model: str, device: torch.device):
        if "resnet" in model:
            from torchvision import transforms
            model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                model,
                pretrained=True,
            )
            # Remove the last linear layer:
            self.model = torch.nn.Sequential(
                *(list(model.children())[:-1])).to(device)
            # Preprocess transformations:
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif "vit" in model:
            from transformers import AutoImageProcessor, ViTModel
            self.preprocess = AutoImageProcessor.from_pretrained(model)
            self.model = ViTModel.from_pretrained(model).to(device)
        self.model.eval()
        self.device = device

    def __call__(self, path_to_images: List[str]) -> Tensor:
        images: list[Image] = []
        for path_to_image in path_to_images:
            try:
                image = Image.open(path_to_image)
            except Exception:
                # There is one image does not exist, thus create a black one:
                image = Image.new("RGB", (600, 471))
            images.append(image.copy())
            image.close()
        images = [image.convert('RGB') for image in images]
        if "ViT" in str(self.preprocess):
            inputs = self.preprocess(images, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            with torch.no_grad():
                res = self.model(**inputs).pooler_output.cpu().detach()
            return res
        else:
            inputs = [self.preprocess(image) for image in images]
            inputs = torch.stack(inputs, dim=0).to(self.device)
            with torch.no_grad():
                res = self.model(inputs).cpu().detach()
            return res.view(len(images), -1)


torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data",
                "diamond_images")
os.makedirs(path, exist_ok=True)

col_to_image_embedder_cfg = ImageEmbedderConfig(
    image_embedder=ImageToEmbedding(args.model, device), batch_size=10)
dataset = DiamondImages(path,
                        col_to_image_embedder_cfg=col_to_image_embedder_cfg)

model_name = args.model.replace('/', '')
filename = f"{model_name}_data.pt"
dataset.materialize(path=osp.join(path, filename))
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset[:0.8], dataset[
    0.8:0.9], dataset[0.9:]

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.image_embedded.parent: LinearEmbeddingEncoder(),
}

model = FTTransformer(
    channels=args.channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        tf = tf.to(device)
        pred = model(tf)
        loss = F.cross_entropy(pred, tf.y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    accum = total_count = 0

    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        pred_class = pred.argmax(dim=-1)
        accum += float((tf.y == pred_class).sum())
        total_count += len(tf.y)

    accuracy = accum / total_count
    return accuracy


metric = "Acc"
best_val_metric = 0
best_test_metric = 0

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
          f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}")

print(f"Best Val {metric}: {best_val_metric:.4f}, "
      f"Best Test {metric}: {best_test_metric:.4f}")
