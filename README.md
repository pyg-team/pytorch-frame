[testing-image]: https://github.com/pyg-team/pytorch-frame/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pyg-team/pytorch-frame/actions/workflows/testing.yml
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/pyg-team/pytorch-frame/blob/master/.github/CONTRIBUTING.md
[slack-image]: https://img.shields.io/badge/slack-pyf-brightgreen
[slack-url]: https://data.pyg.org/slack.html

<p align="center">
  <img height="200" src="https://github.com/pyg-team/pytorch-frame/blob/master/docs/source/_figures/pytorch_frame_logo_text.JPG" />
</p>

--------------------------------------------------------------------------------

[![Testing Status][testing-image]][testing-url]
[![Contributing][contributing-image]][contributing-url]
[![Slack][slack-image]][slack-url]

**[Documentation](https://pyg-team-pytorch-frame.readthedocs-hosted.com)**

PyTorch Frame is a library built upon :pytorch:`null` `PyTorch <https://pytorch.org>`_ to easily write and train tabular deep learning models.

It consists of various methods for deep learning on tables from a variety of published papers. In addition, it consists of easy-to-use mini-batch loaders and a large number of common benchmark datasets. Plus, it features straightforward interfaces for those looking to load their own datasets.

With PyTorch Frame, we aim to democratize the deep learning experience for tabular data. Whether you're an experienced deep learning researcher, a novice delving into machine learning, or a Kaggle enthusiast, PyTorch Frame makes experimenting with different architectures a breeze.

Our goal is to advance deep learning research for tabular data. Historically, tree-based models have superior performance on tabular datasets. However recent research shows that some deep learning models have comparable, if not better, performance on larger datasets, not to say the benefits on training efficiency on large scale data.

* [Library Highlights](#library-highlights)
* [Architecture Overview](#architecture-overview)
* [Implemented Deep Tabular Models](#implemented-deep-tabular-models)

## Library Highlights

Whether you are a machine learning researcher or first-time user of machine learning toolkits, here are some reasons to use PyTorchFrame for deep learning on tabular dataset.

* **Easy-to-use and unified API**:
  PyG is *PyTorch-on-the-rocks*: It utilizes a tensor-centric API and keeps design principles close to vanilla PyTorch.
  If you are already familiar with PyTorch, utilizing PyTorch Frame is straightforward.
* **Comprehensive and well-maintained Deep Tabular Models**:
  Most of the state-of-the-art Deep Tabular architectures have been implemented by library developers and are ready to be applied.
* **PyG Integration**:
  Pytorch Frame can be integrated with PyG. Node feature can be represented with `TensorFrame`, which can include a variety of semantic types, including numerical features, categorical features, text, etc.

## Architecture Overview

Models in PyTorch Frame follow a modular design of `FeatureEncoder`, `TableConv`, and `Decoder`, as shown in the figure below:

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/pyg-team/pytorch-frame/master/docs/source/_figures/modular.png?sanitize=true" />
</p>

## Implemented Deep Tabular Models

We list currently supported deep tabular models:

* **[Trompt](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Chen *et al.*: [Trompt: Towards a Better Deep Neural Network for Tabular Data](https://arxiv.org/abs/2305.18446) (ICML 2023) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/trompt.py)]
* **[FT Transformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[ResNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[TabNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Arık *et al.*: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) (AAAI 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabnet.py)]
* **[ExcelFormer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Chen *et al.*: [ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data](https://arxiv.org/abs/2301.02819) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/excelformer.py)]
* **[TabTransformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Arık *et al.*: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabtransformer.py)]
