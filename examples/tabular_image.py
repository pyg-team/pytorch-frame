import argparse
import os
import os.path as osp

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from torch_frame import stype
from torch_frame.config import ImageEmbedder, ImageEmbedderConfig
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
        "microsoft/resnet-18",
        "google/vit-base-patch16-224-in21k",
        "microsoft/swin-base-patch4-window7-224-in22k",
    ],
)

args = parser.parse_args()

# Image Embedded
# ================ ResNet ===================
# Best Val Acc: 0.2864, Best Test Acc: 0.2789
# ================== ViT ====================
# Best Val Acc: 0.4173, Best Test Acc: 0.4110
# ================= Swin ====================
# Best Val Acc: 0.4345, Best Test Acc: 0.4274


class ImageToEmbedding(ImageEmbedder):
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.model_name = model_name
        self.preprocess = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    def forward_embed(self, images: list[Image]) -> Tensor:
        inputs = self.preprocess(images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            res = self.model(**inputs).pooler_output.cpu().detach()
            if "resnet" in self.model_name:
                res = res.squeeze(dim=(2, 3))
        return res


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
