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
* [Architecture Overview](#architecture-overview)
* [Implemented Deep Tabular Models](#implemented-deep-tabular-models)

## Library Highlights

PyTorch Frame emphasizes a tensor-centric API and maintains design elements similar to vanilla PyTorch. For those acquainted with PyTorch, adapting to PyTorch Frame is a seamless process. Here are our major library highlights:

* **Versitility with Tabular Data**:
  PyTorch Frame provides inhouse support for multimodal learning on a variety of semantic types, including categories, numbers and text.
* **Modular Implementation of Diverse Models**:
  Many existing deep learning models are implemented in a modular way. See next section for [more details](#architecture-overview).
* **Empowerment through Multimodal Learning**:
  PyTorch Frame can mesh with large language models on text, as illustrated in this [example](https://github.com/pyg-team/pytorch-frame/blob/master/examples/fttransformer_text.py).
* **PyG Integration**:
  PyTorch Frame synergizes seamlessly with PyG, enabling `TensorFrame` representation of node features, inclusive of numeric, categorical, textual attributes, and beyond.

## Architecture Overview

Models in PyTorch Frame follow a modular design of `FeatureEncoder`, `TableConv`, and `Decoder`, as shown in the figure below:

<p align="center">
  <img width="100%" src="https://github.com/pyg-team/pytorch-frame/blob/master/docs/source/_figures/modular.png" />
</p>

In essence, this modular setup empowers users to effortlessly experiment with myriad architectures:

* `materialization` stage handles converting the dataset into a `TensorFrame` and computes the column statistics for each semantic type.
* `FeatureEncoder` encodes different semantic types into hidden embeddings.
* `TableConv` handles column-wise interactions between different semantic types.
* `Decoder` summaries the embeddings and generates the prediction outputs.

## Implemented Deep Tabular Models

We list currently supported deep tabular models:

* **[Trompt](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.Trompt.html)** from Chen *et al.*: [Trompt: Towards a Better Deep Neural Network for Tabular Data](https://arxiv.org/abs/2305.18446) (ICML 2023) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/trompt.py)]
* **[FTTransformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.FTTransformer.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[ResNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.ResNet.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[TabNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.TabNet.html)** from Arık *et al.*: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) (AAAI 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabnet.py)]
* **[ExcelFormer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.ExcelFormer.html)** from Chen *et al.*: [ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data](https://arxiv.org/abs/2301.02819) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/excelformer.py)]
* **[TabTransformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.TabTransformer.html)** from Arık *et al.*: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabtransformer.py)]
