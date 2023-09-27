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
:pyg:`PyTorch Frame` contains a large number of common benchmark datasets.
