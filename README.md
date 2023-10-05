[testing-image]: https://github.com/pyg-team/pytorch-frame/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pyg-team/pytorch-frame/actions/workflows/testing.yml
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/pyg-team/pytorch-frame/blob/master/.github/CONTRIBUTING.md
[slack-image]: https://img.shields.io/badge/slack-pyf-brightgreen
[slack-url]: https://data.pyg.org/slack.html

<p align="center">
  <img width="100%" src="https://github.com/pyg-team/pytorch-frame/blob/master/docs/source/_figures/pytorch_frame_logo_text.JPG" />
</p>

--------------------------------------------------------------------------------

[![Testing Status][testing-image]][testing-url]
[![Contributing][contributing-image]][contributing-url]
[![Slack][slack-image]][slack-url]

**[Documentation](https://pyg-team-pytorch-frame.readthedocs-hosted.com)**

PyTorch Frame is a library built upon [PyTorch](https://pytorch.org/) to easily write and train deep learning models on multimodal data tables.

It consists of various methods for deep learning on tables from a variety of published papers. In addition, it consists of easy-to-use mini-batch loaders and a large number of common benchmark datasets. Plus, it features straightforward interfaces for those looking to load their own datasets.

With PyTorch Frame, we aim to democratize the deep learning experience for tabular data. Whether you're an experienced deep learning researcher, a novice delving into machine learning, or a Kaggle enthusiast, PyTorch Frame makes experimenting with different architectures a breeze.

We have two main goals for PyTorch Frame.

One is advance deep learning research for tabular data. Historically, tree-based models have superior performance on tabular datasets. However recent research shows that some deep learning models have comparable, if not better, performance on larger datasets, not to say the benefits of training efficiency on large scale data.

We also want to extend the scope of PyTorch Frame in two dimensions. One is to support more semantic types e.g. time, sequence, SMILES(Simplified Molecular Input Line Entry System) strings. The other is integration with existing technologies like PyG and large language models.

* [Library Highlights](#library-highlights)
* [Architecture Overview](#architecture-overview)
* [Implemented Deep Tabular Models](#implemented-deep-tabular-models)

## Library Highlights

PyTorch Frame emphasizes a tensor-centric API and maintains design elements similar to vanilla PyTorch. For those acquainted with PyTorch, adapting to PyTorch Frame is a seamless process.

* **Easy-to-use on Structured Data with Different Semantic Types**:
  PyTorch Frame provides inhouse support for multimodal learning on a variety of semantic types.
* **Comprehensive and well-maintained Deep Tabular Models**:
  Most of the state-of-the-art deep learning models for tabular data have been implemented by library developers and are ready to be applied.
* **PyG Integration**:
  Pytorch Frame can be integrated with PyG. Node feature can be represented with `TensorFrame`, which can include a variety of semantic types, including numerical features, categorical features, text, etc.

## Architecture Overview

Models in PyTorch Frame follow a modular design of `FeatureEncoder`, `TableConv`, and `Decoder`, as shown in the figure below:

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/pyg-team/pytorch-frame/master/docs/source/_figures/modular.png?sanitize=true" />
</p>

In summary, a tabular dataset is first converted to `TensorFrame`, and then

## Implemented Deep Tabular Models

We list currently supported deep tabular models:

* **[Trompt](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Chen *et al.*: [Trompt: Towards a Better Deep Neural Network for Tabular Data](https://arxiv.org/abs/2305.18446) (ICML 2023) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/trompt.py)]
* **[FT Transformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[ResNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Gorishniy *et al.*: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/revisiting.py)]
* **[TabNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Arık *et al.*: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) (AAAI 2021) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabnet.py)]
* **[ExcelFormer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Chen *et al.*: [ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data](https://arxiv.org/abs/2301.02819) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/excelformer.py)]
* **[TabTransformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_frame.nn.models.SchNet.html)** from Arık *et al.*: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678) [[**Example**](https://github.com/pyg-team/pytorch-frame/blob/master/examples/tabtransformer.py)]
