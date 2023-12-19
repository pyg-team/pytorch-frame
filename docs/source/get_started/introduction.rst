Introduction by Example
=======================

:pyf:`PyTorch Frame` is a tabular deep learning extension library for :pytorch:`null` `PyTorch <https://pytorch.org>`_.
Modern data is stored in a table format with heterogeneous columns each with its own semantic type, *e.g.*, numerical (such as age or price), categorical (such as gender or product type), time, text (such as descriptions or comments), images, etc.
The goal of :pyf:`PyTorch Frame` is to build a deep learning framework to perform effective machine learning on such complex and diverse data.

Many recent tabular models follow the modular design of :obj:`~torch_frame.nn.encoder.FeatureEncoder`, :obj:`~torch_frame.nn.conv.TableConv`, and :obj:`~torch_frame.nn.decoder.Decoder`.
:pyf:`PyTorch Frame` is designed to facilitate the creation, implementation and evaluation of deep learning models for tabular data under such modular architecture.
Please refer to the :doc:`/get_started/modular_design` page for more information.

In this doc, we introduce the fundamental concepts of :pyf:`PyTorch Frame` through self-contained examples.

At its core, :pyf:`PyTorch Frame` provides the following main features:

.. contents::
    :local:

Common Benchmark Datasets
-------------------------

:pyf:`PyTorch Frame` contains a large number of common benchmark datasets. The list
of all datasets are available in :doc:`/modules/datasets`.

Initializing datasets is straightforward in :pyf:`PyTorch Frame`.
An initialization of a dataset will automatically download its raw files and process the columns.

In the below example, we will use one of the pre-loaded datasets, containing the Titanic passengers.
If you would like to use your own dataset, refer to the example in :doc:`/handling_advanced_stypes/handle_heterogeneous_stypes`.

.. code-block:: python

    from torch_frame.datasets import Titanic

    dataset = Titanic(root='/tmp/titanic')

    len(dataset)
    >>> 891

    dataset.feat_cols
    >>> ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    dataset.materialize()
    >>> Titanic()

    dataset.df.head(5)
    >>>
                    Survived  Pclass                                            Name    Sex   Age   SibSp  Parch            Ticket     Fare Cabin Embarked
    PassengerId
    1                   0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
    2                   1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
    3                   1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
    4                   1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
    5                   0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S


:pyf:`PyTorch Frame` also supports a custom dataset, so that you can use :pyf:`PyTorch Frame` for your own problem.
Let's say you prepare your :class:`pandas.DataFrame` as :obj:`df` with five columns:
:obj:`cat1`, :obj:`cat2`, :obj:`num1`, :obj:`num2`, and :obj:`y`.
Creating :obj:`torch_frame.data.Dataset` object is very easy:

.. code-block:: python

    import torch_frame
    from torch_frame.data import Dataset

    # Specify the stype of each column with a dictionary.
    col_to_stype = {"cat1": torch_frame.categorical, "cat2": torch_frame.categorical,
                    "num1": torch_frame.numerical, "num2": torch_frame.numerical,
                    "y": torch_frame.categorical}

    # Set "y" as the target column.
    dataset = Dataset(df, col_to_stype=col_to_stype, target_col="y")

Data Handling of Tables
-----------------------

A table contains different columns with different data types. Each data type is described by a semantic type which we refer to as :class:`~torch_frame.stype`.
Currently :pyf:`PyTorch Frame` supports the following :class:`stypes<torch_frame.stype>`:

- :obj:`stype.categorical<torch_frame.stype>` denotes categorical columns.
- :obj:`stype.numerical<torch_frame.stype>` denotes numerical columns.
- :obj:`stype.multicategorical<torch_frame.stype>` denotes multi_categorical columns.
- :obj:`stype.text_embedded<torch_frame.stype>` denotes text columns that are pre-embedded via some text encoder.

A table in :pyf:`PyTorch Frame` is described by an instance of :class:`~torch_frame.data.TensorFrame`, which holds the following attributes by default:

- :obj:`col_names_dict`: A dictionary holding the column names for each :class:`~torch_frame.stype`.
- :obj:`feat_dict`: A dictionary holding the :obj:`~torch.Tensor` of different :class:`stypes<torch_frame.stype>`.
  For :obj:`stype.numerical<torch_frame.stype>` and :obj:`stype.categorical<torch_frame.stype>`, the shape of :obj:`~torch.Tensor` is [`num_rows`, `num_cols`], while for :obj:`stype.text_embedded<torch_frame.stype>`, the shape is [`num_rows`, `num_cols`, `emb_dim`].
- :obj:`y` (optional): A tensor containing the target values for prediction.

.. note::
    The set of keys in :obj:`feat_dict` must exactly match with the set of keys in :obj:`col_names_dict`.
    :class:`~torch_frame.data.TensorFrame` is validated at initialization time.

Creating a :class:`~torch_frame.data.TensorFrame` from :class:`~torch_frame.data.Dataset` is referred to as materialization.
:meth:`~torch_frame.data.Dataset.materialize` converts raw data frame in :class:`~torch_frame.data.Dataset` into :class:`Tensors<torch.Tensor>` and stores them in a :class:`~torch_frame.data.TensorFrame`.
:meth:`~torch_frame.data.Dataset.materialize` also provides an optional argument `path` to cache the :class:`~torch_frame.data.TensorFrame` and `col_stats`. If `path` is specified,
during the materialization :pyf:`PyTorch Frame` will try to load saved :class:`~torch_frame.data.TensorFrame` and `col_stats` at first. If there is no saved object found for that `path`, :pyf:`PyTorch Frame`
will materialize the dataset and save the materialized :class:`~torch_frame.data.TensorFrame` and `col_stats` to the `path`.

.. note::
    Note that materialization does minimal processing of the original features, e.g., no normalization and missing value handling are performed.
    PyTorch Frame converts missing values in categorical :class:`torch_frame.stype` to `-1` and missing values in numerical :class:`torch_frame.stype` to `NaN`.
    We expect `NaN`/missing-value handling and normalization to be handled by the model side via :class:`torch_frame.nn.encoder.StypeEncoder`.

The :class:`~torch_frame.data.TensorFrame` object has :class:`~torch.Tensor` at its core; therefore, it's friendly for training and inference with PyTorch. In :pyf:`PyTorch Frame`, we build data loaders and models around :class:`~torch_frame.data.TensorFrame`, benefitting from all the efficiency and flexibility from PyTorch.

.. code-block:: python

    from torch_frame import stype

    dataset.materialize() # materialize the dataset

    dataset.materialize(path='/tmp/titanic/data.pt') # materialize the dataset with caching

    dataset.materialize(path='/tmp/titanic/data.pt') # next materialization will load the cache

    tensor_frame = dataset.tensor_frame

    tensor_frame.feat_dict.keys()
    >>> dict_keys([<stype.categorical: 'categorical'>, <stype.numerical: 'numerical'>])

    tensor_frame.feat_dict[stype.numerical]
    >>> tensor([[22.0000,  1.0000,  0.0000,  7.2500],
                [38.0000,  1.0000,  0.0000, 71.2833],
                [26.0000,  0.0000,  0.0000,  7.9250],
                ...,
                [    nan,  1.0000,  2.0000, 23.4500],
                [26.0000,  0.0000,  0.0000, 30.0000],
                [32.0000,  0.0000,  0.0000,  7.7500]])

    tensor_frame.feat_dict[stype.categorical]
    >>> tensor([[0, 0, 0],
                [1, 1, 1],
                [0, 1, 0],
                ...,
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 2]])

    tensor_frame.col_names_dict
    >>> {<stype.categorical: 'categorical'>: ['Pclass', 'Sex', 'Embarked'], <stype.numerical: 'numerical'>: ['Age', 'SibSp', 'Parch', 'Fare']}

    tensor_frame.y
    >>> tensor([0, 1, 1,  ..., 0, 1, 0])

A :class:`~torch_frame.data.TensorFrame` contains the following basic properties:

.. code-block:: python

    tensor_frame.stypes
    >>> [<stype.numerical: 'numerical'>, <stype.categorical: 'categorical'>]

    tensor_frame.num_cols
    >>> 7

    tensor_frame.num_rows
    >>> 891

    tensor_frame.device
    >>> device(type='cpu')


We support transferring the data in a :class:`~torch_frame.data.TensorFrame` to devices supported by :pytorch:`PyTorch`.

.. code-block:: python

    tensor_frame.to("cpu")

    tensor_frame.to("cuda")

Once a :obj:`~torch_frame.data.Dataset` is materialized, we can retrieve column statistics on the data.
For each :class:`~torch_frame.stype`, a different set of statistics is calculated.

For categorical features,

- :class:`StatType.COUNT` contains a tuple of two lists, where first list contains ordered category names and the second list contains category count, sorted from high to low.

For numerical features,

- :class:`StatType.MEAN` denotes the mean value of the numerical feature,
- :class:`StatType.STD` denotes the standard deviation,
- :class:`StatType.QUANTILES` contains a list containing minimum value, first quartile (25th percentile), median (50th percentile), third quartile (75th percentile) and maximum value of the column.

.. code-block:: python

    dataset.col_to_stype
    >>> {'Survived': <stype.categorical: 'categorical'>, 'Pclass': <stype.categorical: 'categorical'>, 'Sex': <stype.categorical: 'categorical'>, 'Age': <stype.numerical: 'numerical'>, 'SibSp': <stype.numerical: 'numerical'>, 'Parch': <stype.numerical: 'numerical'>, 'Fare': <stype.numerical: 'numerical'>, 'Embarked': <stype.categorical: 'categorical'>}

    dataset.col_stats['Sex']
    >>> {<StatType.COUNT: 'COUNT'>: (['male', 'female'], [577, 314])}

    dataset.col_stats['Age']
    >>> {<StatType.MEAN: 'MEAN'>: 29.69911764705882, <StatType.STD: 'STD'>: 14.516321150817316, <StatType.QUANTILES: 'QUANTILES'>: [0.42, 20.125, 28.0, 38.0, 80.0]}

Now let's say you have a new :class:`pandas.DataFrame` called :obj:`new_df`, and
you want to convert it to a corresponding :class:`~torch_frame.data.TensorFrame` object.
You can achieve this as follows:

.. code-block:: python

    new_tf = dataset.convert_to_tensor_frame(new_df)

Mini-batches
------------

Neural networks are usually trained in a mini-batch fashion. :pyf:`PyTorch Frame` contains its own :class:`~torch_frame.data.DataLoader`, which can load :class:`~torch_frame.data.Dataset` or :class:`~torch_frame.data.TensorFrame` in mini batches.

.. code-block:: python

    from torch_frame.data import DataLoader

    data_loader = DataLoader(tensor_frame, batch_size=32,
                            shuffle=True)

    for batch in data_loader:
        batch
        >>> TensorFrame(
                num_cols=7,
                num_rows=32,
                categorical (3): ['Pclass', 'Sex', 'Embarked'],
                numerical (4): ['Age', 'SibSp', 'Parch', 'Fare'],
                has_target=True,
                device='cpu',
            )

Learning Methods on Tabular Data
--------------------------------

After learning about data handling, datasets, and loader in :pyf:`PyTorch Frame`, it’s time to implement our first model!

Now let’s implement a model called :obj:`ExampleTransformer`. It uses :class:`~torch_frame.nn.conv.TabTransformerConv` as its convolution layer.
Initializing a :class:`~torch_frame.nn.encoder.StypeWiseFeatureEncoder` requires :obj:`col_stats` and :obj:`col_names_dict`, we can directly get them as properties of any materialized dataset.

.. code-block:: python

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

In the example above, :class:`~torch_frame.nn.encoder.EmbeddingEncoder` is used to encode the categorical features and
:class:`~torch_frame.nn.encoder.LinearEncoder` is used to encode the numerical features.
The embeddings are then passed into layers of :class:`~torch_frame.nn.conv.TabTransformerConv`.
Then the outputs are concatenated and fed into a :obj:`torch.nn.Linear` decoder.

Let's create train-test split and create data loaders.

.. code-block:: python

    from torch_frame.datasets import Yandex
    from torch_frame.data import DataLoader

    dataset = Yandex(root='/tmp/adult', name='adult')
    dataset.materialize()
    dataset.shuffle()
    train_dataset, test_dataset = dataset[:0.8], dataset[0.80:]
    train_loader = DataLoader(train_dataset.tensor_frame, batch_size=128,
                            shuffle=True)
    test_loader = DataLoader(test_dataset.tensor_frame, batch_size=128)

Let’s train this model for 50 epochs:

.. code-block:: python

    import torch
    import torch.nn.functional as F

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
        for tf in train_loader:
            tf = tf.to(device)
            pred = model(tf)
            loss = F.cross_entropy(pred, tf.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

Finally, we can evaluate our model on the test split:

.. code-block:: python

    model.eval()
    correct = 0
    for tf in test_loader:
        tf = tf.to(device)
        pred = model(tf)
        pred_class = pred.argmax(dim=-1)
        correct += (tf.y == pred_class).sum()
    acc = int(correct) / len(test_dataset)
    print(f'Accuracy: {acc:.4f}')
    >>> Accuracy: 0.8447


This is all it takes to implement your first deep tabular network.
Happy hacking!
