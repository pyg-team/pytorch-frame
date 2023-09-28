Introduction by Example
=======================

We shortly introduce the fundamental concepts of :pyg:`PyTorch Frame` through self-contained examples.

At its core, :pyg:`PyTorch Frame` provides the following main features:

.. contents::
    :local:

Data Handling of Tables
-----------------------
A table contains different columns with different data types. Each data type is described by a semantic type which we refer to as :obj:`stype`.
Currently :pyg:`PyTorch Frame` supports the following :obj:`stype`'s:

- :obj:`stype.categorical` denotes Categorical values,
- :obj:`stype.numerical` denotes Numerical values,
- :obj:`stype.text` denotes Text.

A table in :pyg:`PyTorch Frame` is described by an instance of :class:`TensorFrame`, which holds the following attributes by default:

- :obj:`col_names_dict`: A dictionary holding the column names for each :obj:`stype`.
- :obj:`x_dict`: A dictionary holding the :obj:`Tensor` of different :obj:`stype`'s. The size of :obj:`Tensor` is at least two dimensional with shape [`num_rows`, `num_cols`, \*].
The first dimension represents rows and the second dimension represents columns.
Any remaining dimension describes the feature value of the (row, column) pair.
- :obj:`y`(optional): A tensor containing the target values for prediction.

We show a simple example of a table with 3 categorical columns and 2 numerical columns.

.. code-block:: python

    from torch_frame import stype
    from torch_frame import TensorFrame

    num_rows = 10

    x_dict = {
        torch_frame.categorical: torch.randint(0, 3, size=(num_rows, 3)),
        torch_frame.numerical: torch.randn(size=(num_rows, 2)),
    }

    col_names_dict = {
        torch_frame.categorical: ['a', 'b', 'c'],
        torch_frame.numerical: ['x', 'y'],
    }

    y = torch.randn(num_rows)

    tensor_frame = TensorFrame(
            x_dict=x_dict,
            col_names_dict=col_names_dict,
            y=y,
        )

    >>> TensorFrame(
            num_cols=5,
            num_rows=10,
            categorical (3): ['a', 'b', 'c'],
            numerical (2): ['x', 'y'],
            has_target=True,
            device=cpu,
        )

.. note::
    The set of keys in `x_dict` must exactly match with the set of keys in `col_names_dict`.
    We validate the :obj:`TensorFrame` at initialization time.

A :obj:`TensorFrame` contains many properties.

.. code-block:: python

    tensor_frame.stypes
    >>> [<stype.numerical: 'numerical'>, <stype.categorical: 'categorical'>]

    tensor_frame.num_cols
    >>> 5

    tensor_frame.num_rows
    >>> 10

    tensor_frame.device
    >>> device(type='cpu')

We support transferring the data in a :obj:`TensorFrame` across devices.

.. code-block:: python

    tensor_frame.cpu()

    tensor_frame.cuda()

Common Benchmark Datasets
-------------------------
:pyg:`PyTorch Frame` contains a large number of common benchmark datasets, *e.g.*, datasets from `https://github.com/yandex-research/tabular-dl-revisiting-models <https://github.com/yandex-research/tabular-dl-revisiting-models>`_
, datasets from `tabular benchmark <https://huggingface.co/datasets/inria-soda/tabular-benchmark>`_ .

Initializing datasets is straightforward in :pyg:`PyTorch Frame`. An initialization of a dataset will automatically download its raw files and process the columns, *e.g*., to load the bank-marketing dataset, type:

.. code-block:: python

    from torch_frame.datasets import Yandex

    dataset = Yandex(root='/tmp/adult', name='adult')

    len(dataset)
    >>> 48842

    dataset.feat_cols
    >>> ['C_feature_0', 'C_feature_1', 'C_feature_2', 'C_feature_3', 'C_feature_4', 'C_feature_5', 'C_feature_6', 'C_feature_7', 'N_feature_0', 'N_feature_1', 'N_feature_2', 'N_feature_3', 'N_feature_4', 'N_feature_5']

We can use slices, long or bool tensors to split the dataset, *e.g.*, to create a 90/10 train/test split, type:

.. code-block:: python

    train_dataset = dataset[:0.9]
    >>> Yandex()

    len(train_dataset)
    >>> 43958

    test_dataset = dataset[0.9:]
    >>> Yandex()

    len(test_dataset)
    >>> 4884

If you are unsure whether the dataset is already shuffled before you split, you can randomly permutate it by running:

.. code-block:: python

    dataset.shuffle(return_perm=True)
    >>> (Yandex(), tensor([40091, 36301, 47858,  ...,  2003, 11049, 25131]))

This is equivalent of doing:

.. code-block:: python

    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]


Converting a :class:`torch_frame.dataset.Dataset` into a :obj:`TensorFrame` instance refers to a materialization stage from raw data into compact :obj:`Tensor` representations.
We show a simple example.

.. code-block:: python

    dataset.materialize() # materialize the dataset

    tensor_frame = dataset.tensor_frame

    tensor_frame.col_names_dict
    >>> {<stype.categorical: 'categorical'>: ['C_feature_0', 'C_feature_1', 'C_feature_2', 'C_feature_3', 'C_feature_4', 'C_feature_5', 'C_feature_6', 'C_feature_7'], <stype.numerical: 'numerical'>: ['N_feature_0', 'N_feature_1', 'N_feature_2', 'N_feature_3', 'N_feature_4', 'N_feature_5']}

    tensor_frame.y
    >>> tensor([0, 0, 0,  ..., 0, 0, 1])

Once a :obj:`torch_frame.dataset.Dataset` is materialized, we can retrieve column statistics on the data.

For each :obj:`stype`, a different set of statistics is calculated.

For categorical features,

- :obj:`StatType.COUNT` contains a tuple of two list, where first list contains ordered category names and the second list contains category count, sorted from high to low.

For numerical features,

- :obj:`StatType.MEAN` denotes the mean value of the numerical feature,
- :obj:`StatType.STD` denotes the standard deviation,
- :obj:`StatType.QUANTILES` contains a list containing minimum value, first quartile(25th percentile), median(50th percentile), thrid quartile(75th percentile) and maximum value of the column.

.. code-block:: python

    dataset.col_to_stype
    >>> {'C_feature_0': <stype.categorical: 'categorical'>, 'C_feature_1': <stype.categorical: 'categorical'>, 'C_feature_2': <stype.categorical: 'categorical'>, 'C_feature_3': <stype.categorical: 'categorical'>, 'C_feature_4': <stype.categorical: 'categorical'>, 'C_feature_5': <stype.categorical: 'categorical'>, 'C_feature_6': <stype.categorical: 'categorical'>, 'C_feature_7': <stype.categorical: 'categorical'>, 'N_feature_0': <stype.numerical: 'numerical'>, 'N_feature_1': <stype.numerical: 'numerical'>, 'N_feature_2': <stype.numerical: 'numerical'>, 'N_feature_3': <stype.numerical: 'numerical'>, 'N_feature_4': <stype.numerical: 'numerical'>, 'N_feature_5': <stype.numerical: 'numerical'>, 'label': <stype.categorical: 'categorical'>}

    dataset.col_stats['C_feature_0']
    >>> {<StatType.COUNT: 'COUNT'>: (['Private', 'Self-emp-not-inc', 'Local-gov', 'nan', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked'], [33906, 3862, 3136, 2799, 1981, 1695, 1432, 21, 10])}

    dataset.col_stats['N_feature_0']
    >>> {<StatType.MEAN: 'MEAN'>: 38.64358543876172, <StatType.STD: 'STD'>: 13.71036957798689, <StatType.QUANTILES: 'QUANTILES'>: [17.0, 28.0, 37.0, 48.0, 90.0]}

Mini-batches
-----------------------
Neural networks are usually trained in a batch-wise fashion. :pyg:`PyTorch Frame` contains its own :obj:`torch_frame.data.DataLoader`, which can load :obj:`torch_frame.data.Dataset` or :obj:`TensorFrame` in mini batches.

.. code-block:: python

    from torch_frame.data import DataLoader

    data_loader = DataLoader(tensor_frame, batch_size=32,
                            shuffle=True)

    for batch in loader:
        batch
        >>> TensorFrame(
                num_cols=14,
                num_rows=32,
                categorical (8): ['C_feature_0', 'C_feature_1', 'C_feature_2', 'C_feature_3', 'C_feature_4', 'C_feature_5', 'C_feature_6', 'C_feature_7'],
                numerical (6): ['N_feature_0', 'N_feature_1', 'N_feature_2', 'N_feature_3', 'N_feature_4', 'N_feature_5'],
                has_target=True,
                device=cpu,
            )

Data Transforms
-----------------------
:pyg:`PyTorch Frame` allows for data transformation across different :obj:`stype`'s or within the same :obj:`stype`. Transforms takes in both :obj:`TensorFrame` and column stats.

Let's look an example, where we apply `CatBoostEncoder <https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-to-numberic>` to transform the categorical features into numerical features.

.. code-block:: python

    from torch_frame.datasets import Yandex
    from torch_frame.transforms import CategoricalCatBoostEncoder

    dataset = Yandex(root='/tmp/adult', name='adult')
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    transform = CategoricalCatBoostEncoder()
    transform.fit(tensor_frame, dataset.col_stats)

    transformed_col_stats = transform.transformed_stats
    transformed_col_stats['C_feature_0']
    >>> {<StatType.MEAN: 'MEAN'>: 0.23928034419669833, <StatType.STD: 'STD'>: 0.07742150848292455, <StatType.QUANTILES: 'QUANTILES'>: [0.021752887790314594, 0.21786767575325724, 0.21786767575325724, 0.21786767575325724, 0.5532071236826023]}

    transform(tensor_frame)
    >>> TensorFrame(
            num_cols=14,
            num_rows=48842,
            numerical (14): ['N_feature_0', 'N_feature_1', 'N_feature_2', 'N_feature_3', 'N_feature_4', 'N_feature_5', 'C_feature_0', 'C_feature_1', 'C_feature_2', 'C_feature_3', 'C_feature_4', 'C_feature_5', 'C_feature_6', 'C_feature_7'],
            has_target=True,
            device=cpu,
        )
