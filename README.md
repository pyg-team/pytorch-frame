[testing-image]: https://github.com/pyg-team/pytorch-frame/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pyg-team/pytorch-frame/actions/workflows/testing.yml
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/pyg-team/pytorch-frame/blob/master/.github/CONTRIBUTING.md
[slack-image]: https://img.shields.io/badge/slack-pyf-brightgreen
[slack-url]: https://data.pyg.org/slack.html

<div align="center">

<img height="175" src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pytorch_frame_logo_text.png?sanitize=true" />

<br>
<br>

**A modular deep learning framework for building neural network models on heterogeneous tabular data.**

--------------------------------------------------------------------------------

[![Testing Status][testing-image]][testing-url]
[![Contributing][contributing-image]][contributing-url]
[![Slack][slack-image]][slack-url]

</div>

**[Documentation](https://pytorch-frame.readthedocs.io)**

PyTorch Frame is a deep learning extension for [PyTorch](https://pytorch.org/), designed for heterogeneous tabular data with different column types, including numerical, categorical, time, text, and images. It offers a modular framework for implementing existing and future methods. The library features methods from state-of-the-art models, user-friendly mini-batch loaders, benchmark datasets, and interfaces for custom data integration.

PyTorch Frame democratizes deep learning research for tabular data, catering to both novices and experts alike. Our goals are:

1. **Facilitate Deep Learning for Tabular Data:** Historically, tree-based models (e.g., XGBoost) excelled at tabular learning but had notable limitations, such as integration difficulties with downstream models such as GNNs, and handling complex column types (e.g., text, sequences, images). Deep tabular models are promising to resolve the limitations.

2. **Expand Functionalities and Model Architectures:** We are enhancing PyTorch Frame to handle diverse column types, like time images, language, and sequences, and integrate cutting-edge technologies like large language models.

* [Library Highlights](#library-highlights)
* [Architecture Overview](#architecture-overview)
* [Quick Tour](#quick-tour)
* [Implemented Deep Tabular Models](#implemented-deep-tabular-models)
* [Benchmark](#benchmark)
* [Installation](#installation)

## Library Highlights

PyTorch Frame builds directly upon PyTorch, ensuring a smooth transition for existing PyTorch users. Key features include:

* **Diverse column types**:
  Supports learning across various column types like categorical, numberical, and texts. Future plans encompass sequences, multicategories, images, and time.
* **Modular model design**:
  Enables modular deep learning model implementations, promoting reusability, clear coding, and experimentation flexibility. Further details in the [architecture overview](#architecture-overview).
* **Models**
  Implements many [state-of-the-art deep tabular models](#implemented-deep-tabular-models) as well as strong GBDTs (XGBoost and CatBoost) with hyper-parameter tuning.
* **Datasets**:
  Comes with a collection of readily-usable tabular datasets. Also supports custom datasets to solve your own problem.
  We [benchmark](https://github.com/pyg-team/pytorch-frame/blob/master/benchmark) deep tabular models against GBDTs.
* **Pytorch integration**:
  Integrates effortlessly with other PyTorch libraries, like [PyG](https://pyg.org/), facilitating end-to-end training of PyTorch Frame with downstream PyTorch models.

## Architecture Overview

Models in PyTorch Frame follow a modular design of `FeatureEncoder`, `TableConv`, and `Decoder`, as shown in the figure below:

<p align="center">
  <img width="100%" src="https://github.com/pyg-team/pytorch-frame/blob/master/docs/source/_figures/modular.png" />
</p>

In essence, this modular setup empowers users to effortlessly experiment with myriad architectures:

* `Materialization` handles converting the raw pandas `DataFrame` into a `TensorFrame` that is amenable to Pytorch-based training and modeling.
* `FeatureEncoder` encodes `TensorFrame` into hidden column embeddings of size `[batch_size, num_cols, channels]`.
* `TableConv` models column-wise interactions over the hidden embeddings.
* `Decoder` generates embedding/prediction per row.


## Quick Tour

In this quick tour, we showcase the ease of creating and training a deep tabular model with only a few lines of code.

### Build your own deep tabular model

In the first example, we implement a simple `ExampleTransformer` following the modular architecture of Pytorch Frame. A model maps `TensorFrame` into embeddings. We decompose `ExampleTransformer`, and most other models in Pytorch Frame into three modular components.

* `self.encoder`: The encoder maps an input `TensorFrame` to an embedding of size `[batch_size, num_cols, channels]`. To handle input of different column types, we use `StypeWiseFeatureEncoder` where users can specify different encoders using a dictionary. In this example, we use `EmbeddingEncoder` for categorical features and `LinearEncoder` for numerical features--they are both built-in encoders in Pytorch Frame.
* `self.convs`: We create a two layers of `TabTransformerConv`. Each `TabTransformerConv` module transforms an embedding of size `[batch_size, num_cols, channels]` and into an embedding of the same size.
* `self.decoder`: We use a mean-based decoder that maps the dimension of the embedding back from `[batch_size, num_cols, channels]` to `[batch_size, out_channels]`.

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
<summary>Once we decide the model, we can load the Adult Census Income dataset and create a train dataloader.</summary>

```python
    from torch_frame.datasets import Yandex

    dataset = Yandex(root='/tmp/adult', name='adult')
    dataset.materialize()
    train_dataset = dataset[:0.8]
    train_loader = DataLoader(train_dataset.tensor_frame, batch_size=128,
                            shuffle=True)
```
</details>

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

## Implemented Deep Tabular Models

We list currently supported deep tabular models:

* **[Trompt](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.Trompt.html)** from Chen *et al.*: [Trompt: Towards a Better Deep Neural Network for Tabular Data](https://arxiv.org/abs/2305.18446) (ICML 2023) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/trompt.py)]
* **[FTTransformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.FTTransformer.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[ResNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.ResNet.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[TabNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.TabNet.html)** from Arık *et al.*: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) (AAAI 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabnet.py)]
* **[ExcelFormer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.ExcelFormer.html)** from Chen *et al.*: [ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data](https://arxiv.org/abs/2301.02819) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/excelformer.py)]
* **[TabTransformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.TabTransformer.html)** from Huang *et al.*: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabtransformer.py)]

In addition, we implemented `XGBoost` and `CatBoost` [examples](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tuned_gbdt.py) with hyperparameter-tuning using [Optuna](https://optuna.org/) for users who'd like to compare their model performance with `GBDTs`.


## Benchmark

We benchmark recent tabular deep learning models against GBDTs over diverse public datasets with different sizes and task types.

The following chart shows the performance of various deep learning models on small regression datasets, where the row represents the model names and the column represents dataset indices (we have 13 datasets here). For more results on classification and larger datasets, please check the [benchmark documentation](https://github.com/pyg-team/pytorch-frame/blob/master/benchmark).

|                     | dataset_0               | dataset_1               | dataset_2               | dataset_3               | dataset_4               | dataset_5               | dataset_6               | dataset_7               | dataset_8               | dataset_9               | dataset_10              | dataset_11              | dataset_12              |
|:--------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| XGBoost             | **0.247±0.000** | 0.077±0.000     | 0.167±0.000     | 1.119±0.000     | 0.328±0.000     | 1.024±0.000     | **0.292±0.000** | 0.606±0.000     | **0.876±0.000** | 0.023±0.000     | **0.697±0.000** | **0.865±0.000** | 0.435±0.000     |
| CatBoost            | 0.265±0.000     | 0.062±0.000     | 0.128±0.000     | 0.336±0.000     | 0.346±0.000     | 0.443±0.000     | 0.375±0.000     | **0.273±0.000** | 0.881±0.000     | 0.040±0.000     | 0.756±0.000     | 0.876±0.000     | 0.439±0.000     |
| Trompt              | 0.261±0.003     | **0.015±0.005** | **0.118±0.001** | **0.262±0.001** | **0.323±0.001** | 0.418±0.003     | 0.329±0.009     | 0.312±0.002     | OOM             | **0.008±0.001** | 0.779±0.006     | 0.874±0.004     | **0.424±0.005** |
| ResNet              | 0.288±0.006     | 0.018±0.003     | 0.124±0.001     | 0.268±0.001     | 0.335±0.001     | 0.434±0.004     | 0.325±0.012     | 0.324±0.004     | 0.895±0.005     | 0.036±0.002     | 0.794±0.006     | 0.875±0.004     | 0.468±0.004     |
| FTTransformerBucket | 0.325±0.008     | 0.096±0.005     | 0.360±0.354     | 0.284±0.005     | 0.342±0.004     | 0.441±0.003     | 0.345±0.007     | 0.339±0.003     | OOM             | 0.105±0.011     | 0.807±0.010     | 0.885±0.008     | 0.468±0.006     |
| ExcelFormer         | 0.302±0.003     | 0.099±0.003     | 0.145±0.003     | 0.382±0.011     | 0.344±0.002     | **0.411±0.005** | 0.359±0.016     | 0.336±0.008     | OOM             | 0.192±0.014     | 0.794±0.005     | 0.890±0.003     | 0.445±0.005     |
| FTTransformer       | 0.335±0.010     | 0.161±0.022     | 0.140±0.002     | 0.277±0.004     | 0.335±0.003     | 0.445±0.003     | 0.361±0.018     | 0.345±0.005     | OOM             | 0.106±0.012     | 0.826±0.005     | 0.896±0.007     | 0.461±0.003     |
| TabNet              | 0.279±0.003     | 0.224±0.016     | 0.141±0.010     | 0.275±0.002     | 0.348±0.003     | 0.451±0.007     | 0.355±0.030     | 0.332±0.004     | 0.992±0.182     | 0.015±0.002     | 0.805±0.014     | 0.885±0.013     | 0.544±0.011     |
| TabTransformer      | 0.624±0.003     | 0.229±0.003     | 0.369±0.005     | 0.340±0.004     | 0.388±0.002     | 0.539±0.003     | 0.619±0.005     | 0.351±0.001     | 0.893±0.005     | 0.431±0.001     | 0.819±0.002     | 0.886±0.005     | 0.545±0.004     |


We see that some recent deep tabular models were able to achieve competitive model performance to strong GBDTs (despite being 5--100 times slower to train). Making deep tabular models even more performant with less compute is a fruitful direction of future research.

## Installation

PyTorch Frame is available for Python 3.8 to Python 3.11.

```
pip install pytorch_frame
```

See [the installation guide](https://pyg-team-pytorch-frame.readthedocs.build/en/latest/get_started/installation.html) for other options.
