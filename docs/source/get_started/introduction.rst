Introduction by Example
=======================

We shortly introduce the fundamental concepts of :pyg:`PyTorch Frame` through self-contained examples.

At its core, :pyg:`PyTorch Frame` provides the following main features:

.. contents::
    :local:

Data Types
-----------------------
A table contains different columns containing different data types. Each data type is described by a :obj:`stype`. Currently :pyg:`PyTorch Frame` supports the following :obj:`stype`'s:

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

TensorFrames
-----------------------
A table's columns data can be represented with a :obj:`TensorFrame`, which holds the following attributes by default:

- :obj:`col_names_dict`: A dictionary holding the column names of different :obj:`stypes`.
- :obj:`x_dict`: A dictionary holding the :obj:`Tensor` of different :obj:`stypes`. The :obj:`Tensor` is a two-dimentional :obj:`Tensor`. The first dimension represents rows and the second dimension represent columns.

We show a simple example of a :obj:`TensorFrame`.

.. code-block:: python
    tensor_frame = train_dataset.tensor_frame

    tensor_frame.col_names_dict
>>> {<stype.categorical: 'categorical'>: ['C_feature_0', 'C_feature_1', 'C_feature_2', 'C_feature_3', 'C_feature_4', 'C_feature_5', 'C_feature_6', 'C_feature_7'], <stype.numerical: 'numerical'>: ['N_feature_0', 'N_feature_1', 'N_feature_2', 'N_feature_3', 'N_feature_4', 'N_feature_5']}
    tensor_frame.x_dict
>>> {<stype.categorical: 'categorical'>: tensor([[3, 1, 1,  ..., 0, 1, 0],
        [0, 1, 0,  ..., 0, 0, 0],
        [0, 1, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 1,  ..., 0, 0, 0],
        [3, 9, 0,  ..., 0, 0, 0],
        [0, 2, 0,  ..., 0, 0, 0]]), <stype.numerical: 'numerical'>: tensor([[1.9000e+01, 1.4040e+05, 1.0000e+01, 0.0000e+00, 0.0000e+00, 3.0000e+01],
        [5.0000e+01, 1.5828e+05, 1.0000e+01, 0.0000e+00, 0.0000e+00, 4.0000e+01],
        [6.2000e+01, 1.8374e+05, 1.0000e+01, 0.0000e+00, 0.0000e+00, 4.0000e+01],
        ...,
        [2.1000e+01, 2.0576e+05, 9.0000e+00, 0.0000e+00, 0.0000e+00, 4.0000e+01],
        [7.3000e+01, 1.9139e+05, 1.5000e+01, 0.0000e+00, 0.0000e+00, 4.0000e+01],
        [5.3000e+01, 3.1135e+05, 1.3000e+01, 0.0000e+00, 0.0000e+00, 5.0000e+01]])}
