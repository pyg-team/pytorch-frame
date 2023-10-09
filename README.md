[testing-image]: https://github.com/pyg-team/pytorch-frame/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pyg-team/pytorch-frame/actions/workflows/testing.yml
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/pyg-team/pytorch-frame/blob/master/.github/CONTRIBUTING.md
[slack-image]: https://img.shields.io/badge/slack-pyf-brightgreen
[slack-url]: https://data.pyg.org/slack.html
<div align="center">

<img width="800px" style="max-width: 100%;" src="https://github.com/pyg-team/pytorch-frame/blob/master/docs/source/_figures/pytorch_frame_logo_text.JPG" />

<br/>
<br/>

**A modular deep learning framework for building neural network models on heterogeneous tabular data.**

--------------------------------------------------------------------------------

[![Testing Status][testing-image]][testing-url]
[![Contributing][contributing-image]][contributing-url]
[![Slack][slack-image]][slack-url]

</div>

**[Documentation](https://pyg-team-pytorch-frame.readthedocs-hosted.com)**

PyTorch Frame is a tabular deep learning extension library for [PyTorch](https://pytorch.org/). Modern data is stored in a table format with heterogeneous columns with different semantic types, e.g. numerical (age, price), categorical (gender, product type), time, text(descriptions), images(pictures), etc. The goal of PyTorch Frame is to build a deep learning framework to perform effective machine learning on such complex data.

PyTorch Frame allow existing (and future) methods to be easily and intuitively implemented in a modular way. The library includes various methods for deep learning on tables from a variety of published papers. In addition, it includes easy-to-use mini-batch loaders, a large number of common benchmark datasets, and intuitive interfaces for custom dataset integration.

With PyTorch Frame, we aim to democratize the deep learning research for tabular data. Whether you're an experienced deep learning researcher, a novice delving into machine learning, or a Kaggle enthusiast, PyTorch Frame makes experimenting with different architectures a breeze.

Our aspirations for PyTorch Frame are twofold:

1. **To Propel Deep Learning Research for Tabular Data:** Historically, tree-based models have superior performance on tabular datasets. However, tree-based models have many limitations, for example, they cannot be trained with downstream models like GNNs, RNNs and Transformers, hence hard to be integrated into larger systems. Tree-based models also cannot handle diverse column types, like text or sequences. Recent research shows that some deep learning models have comparable, if not better, performance on larger datasets. This is not to mention the advantages in training efficiency with massive data scales.

2. **To Support Enhanced Semantic Types and Model Architectures:** We aim to extend PyTorch Frame's functionalities to handle a wider variety of semantic types, such as time sequences. Concurrently, we're focusing on extending PyTorch Frame to latest technologies like large language models.

* [Library Highlights](#library-highlights)
* [Quick Tour](#quick-tour)
* [Architecture Overview](#architecture-overview)
* [Implemented Deep Tabular Models](#implemented-deep-tabular-models)
* [Installation](#installation)

## Library Highlights

PyTorch Frame emphasizes a tensor-centric API and maintains design elements similar to vanilla PyTorch. For those acquainted with PyTorch, adapting to PyTorch Frame is a seamless process. Here are our major library highlights:

* **Versitility with Tabular Data**:
  PyTorch Frame provides in-house support for multimodal learning on a variety of semantic types, including categories, numbers and texts. Extensions to other types, including but not limited to sequences, multicategoricies, images, time are on our road map.
* **Modular Implementation of Diverse Models**:
  We provide a framework for users to implement deep learning models in a modular way, enhancing module reusability, code clarity and ease of experimentation. See next section for [more details](#architecture-overview).
* **Empowerment through Multimodal Learning**:
  PyTorch Frame can mesh with a variety of different transformers on different semantic types, e.g. large language models on text, as illustrated in this [example](https://github.com/pyg-team/pytorch-frame/blob/master/examples/fttransformer_text.py).
* **PyG Integration**:
  PyTorch Frame synergizes seamlessly with PyG, enabling `TensorFrame` representation of node features, inclusive of numeric, categorical, textual attributes, and beyond.

## Quick Tour

In this quick tour, we showcase the ease of creating and training a deep tabular model with only a few lines of code.

### Build your own deep tabular model

In the first example, we implement a simple `ExampleTransformer` following the modular architecture of Pytorch Frame. A model maps `TensorFrame` into embeddings. We decompose `ExampleTransformer`, and most other models in Pytorch Frame into three modular components.

* `self.encoder`: The encoder maps an input `tensor` of size `[batch_size, num_cols]` to an embedding of size `[batch_size, num_cols, channels]`. To handle input of different semantic types, we use `StypeWiseFeatureEncoder` where users can specify different encoders using a dictionary. In this example, we use `EmbeddingEncoder` for categorical features and `LinearEncoder` for numerical features--they are both built-in encoders in Pytorch Frame. For a comprehensive list, check out this [file](https://github.com/pyg-team/pytorch-frame/blob/master/torch_frame/nn/encoder/stype_encoder.py).
* `self.convs`: The convolution handles column-wise interactions. In the example we create a two layers of `TabTransformerConv`, taken from the `TabTransformer` model.
* `self.decoder`: The decoder converts embeddings to prediction outputs. In the example, we use a mean-based decoder that maps the dimension of the embedding back to `[batch_size, out_channels]`.

```python
from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Linear, Module, ModuleList

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import TabTransformerConv
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
)


class ExampleTransformer(Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder()
            },
        )
        self.tab_transformer_convs = ModuleList([
            TabTransformerConv(
                channels=channels,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])
        self.decoder = Linear(channels, out_channels)

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        for tab_transformer_conv in self.tab_transformer_convs:
            x = tab_transformer_conv(x)
        out = self.decoder(x.mean(dim=1))
        return out
```

<details>
<summary>We can now optimize the model in a training loop, similar to the <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation">standard PyTorch training procedure</a>.</summary>

```python
import torch
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ExampleTransformer(
    channels=32,
    out_channels=dataset.num_classes,
    num_layers=2,
    num_heads=8,
    col_stats=train_dataset.col_stats,
    col_names_dict=train_dataset.tensor_frame.col_names_dict,
).to(device)

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(50):
    for tf in tqdm(train_loader):
        pred = model.forward(tf)
        loss = F.cross_entropy(pred, tf.y)
        optimizer.zero_grad()
        loss.backward()
```
</details>

## Architecture Overview

Models in PyTorch Frame follow a modular design of `FeatureEncoder`, `TableConv`, and `Decoder`, as shown in the figure below:

<p align="center">
  <img width="100%" src="https://github.com/pyg-team/pytorch-frame/blob/master/docs/source/_figures/modular.png" />
</p>

In essence, this modular setup empowers users to effortlessly experiment with myriad architectures:

* `Materialization` handles converting the dataset into a `TensorFrame` and computes the column statistics for each semantic type.
* `FeatureEncoder` encodes different semantic types into hidden embeddings.
* `TableConv` handles column-wise interactions between different semantic types.
* `Decoder` summarizes the embeddings and generates the prediction outputs.

## Implemented Deep Tabular Models

We list currently supported deep tabular models:

* **[Trompt](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.Trompt.html)** from Chen *et al.*: [Trompt: Towards a Better Deep Neural Network for Tabular Data](https://arxiv.org/abs/2305.18446) (ICML 2023) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/trompt.py)]
* **[FTTransformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.FTTransformer.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[ResNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.ResNet.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[TabNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.TabNet.html)** from Arık *et al.*: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) (AAAI 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabnet.py)]
* **[ExcelFormer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.ExcelFormer.html)** from Chen *et al.*: [ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data](https://arxiv.org/abs/2301.02819) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/excelformer.py)]
* **[TabTransformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.TabTransformer.html)** from Huang *et al.*: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabtransformer.py)]

In addition, we implemented `XGBoost` and `CatBoost` [examples](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tuned_gbdt.py) with hyperparameter-tuning using [Optuna](https://optuna.org/) for users who'd like to compare their model performance with `GBDTs`.


## Installation

To be added when we cut the first release version.
