Introduction by Example
=======================

We shortly introduce the fundamental concepts of :pyg:`PyTorch Frame` through self-contained examples.

At its core, :pyg:`PyTorch Frame` provides the following main features:

.. contents::
    :local:

Data Handling of Tabular Columns
-----------------------
A table contains different columns containing different data types. Each data type is described by a :obj:`stype`. Currently :pyg:`PyTorch Frame` supports the following :obj:`stype`'s':
- :obj:`stype.categorical`: Categorical features
- :obj:`stype.numerical`: Numerical features
- :obj:`stype.text`: text

Common Benchmark Datasets
-------------------------
:pyg:`PyTorch Frame` contains a large number of common benchmark datasets, *e.g.*, datasets from `https://github.com/yandex-research/tabular-dl-revisiting-models <https://github.com/yandex-research/tabular-dl-revisiting-models>`.

Initializing the dataset is straightforward. An initialization of a dataset will automatically download its raw files and process them to the previously described Dataset format. E.g., to load the adult dataset, type:

.. code-block:: python

    from torch_frame.datasets import Yandex

    dataset = Yandex(root='/tmp/adult', name='adult')
    >>> Downloading https://data.pyg.org/datasets/tables/revisiting_data/adult.zip

    len(dataset)
    >>> 48842

    dataset.materialize()

    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')
    test_dataset = dataset.get_split_dataset('test')
