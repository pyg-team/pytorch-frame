Introduction by Example
=======================

:pyg:`PyTorch Frame` is a tabular deep learning extension library for :pytorch:`null` `PyTorch <https://pytorch.org>`_.
Many recent tabular models follow the modular design of encoders, convolution and decoders.
:pyg:`PyTorch Frame` is designed to facilitate the creation, implementation and evaluation of deep learning models for tabular data under such modular architecture.
Please refer to the :doc:`/get_started/modular_design` page for more information.

In this doc, we shortly introduce the fundamental concepts of :pyg:`PyTorch Frame` through self-contained examples.

At its core, :pyg:`PyTorch Frame` provides the following main features:

.. contents::
    :local:

Common Benchmark Datasets
-------------------------
:pyg:`PyTorch Frame` contains a large number of common benchmark datasets, *e.g.*, datasets from `https://github.com/yandex-research/tabular-dl-revisiting-models <https://github.com/yandex-research/tabular-dl-revisiting-models>`_
, datasets from `tabular benchmark <https://huggingface.co/datasets/inria-soda/tabular-benchmark>`_ .

Initializing datasets is straightforward in :pyg:`PyTorch Frame`.
An initialization of a dataset will automatically download its raw files and process the columns, *e.g*., to load the `Adult Census Income` dataset, type:

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

.. note::
    Each dataset contains only one table.

Data Handling of Tables
-----------------------
A table contains different columns with different data types. Each data type is described by a semantic type which we refer to as :obj:`stype`.
Currently :pyg:`PyTorch Frame` supports the following :obj:`stype`'s:

- :obj:`stype.categorical` denotes categorical values.
- :obj:`stype.numerical` denotes numerical values.
- :obj:`stype.text` denotes text.

A table in :pyg:`PyTorch Frame` is described by an instance of :class:`TensorFrame`, which holds the following attributes by default:

- :obj:`col_names_dict`: A dictionary holding the column names for each :obj:`stype`.
- :obj:`feat_dict`: A dictionary holding the :obj:`Tensor` of different :obj:`stype`'s.
The size of :obj:`Tensor` is at least two dimensional with shape [`num_rows`, `num_cols`, \*]. The first dimension represents rows and the second dimension represents columns. Any remaining dimension describes the feature value of the (row, column) pair.
- :obj:`y`(optional): A tensor containing the target values for prediction.

We show a simple example of a table with 3 categorical columns and 2 numerical columns.

.. code-block:: python

    from torch_frame import stype
    from torch_frame import TensorFrame

    num_rows = 10

    feat_dict = {
        torch_frame.categorical: torch.randint(0, 3, size=(num_rows, 3)),
        torch_frame.numerical: torch.randn(size=(num_rows, 2)),
    }

    col_names_dict = {
        torch_frame.categorical: ['a', 'b', 'c'],
        torch_frame.numerical: ['x', 'y'],
    }

    y = torch.randn(num_rows)

    tensor_frame = TensorFrame(
            feat_dict=feat_dict,
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
    When a :obj:`TensorFrame` is initialized, the data in each categorical column is transformed into index-based integers from [0,`num_categories`-1].
    The categories are sorted by their frequencies in descending order, mapping each category to their rank in sorted order.
    Any invalid entries within the categorical columns are assigned to a value of -1.



.. note::
    The set of keys in `feat_dict` must exactly match with the set of keys in `col_names_dict`.
    :obj:`TensorFrame` is validated at initialization time.

A :obj:`TensorFrame` contains many properties:

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
------------
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

Learning Methods on Tabular Data
--------------------------------

After learning about data handling, datasets and loader in :pyg:`PyTorch Frame`, it’s time to implement our first model!

.. code-block:: python

    from torch_frame.datasets import Yandex

    dataset = Yandex(root='/tmp/adult', name='adult')
    dataset.materialize()

Now let’s implement a simplified version of TabTransformer:

.. code-block:: python

    from typing import Any, Dict, List
    import torch
    from torch import Tensor
    from torch.nn import (
            LayerNorm,
            Linear,
            Module,
            ModuleList,
            ReLU,
            Sequential,
    )

    import torch_frame
    from torch_frame import stype, TensorFrame
    from torch_frame.data.stats import StatType
    from torch_frame.nn import EmbeddingEncoder
    from torch_frame.nn.conv import TabTransformerConv

    class TabTransformer(Module):
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
            categorical_col_len = len(col_names_dict[stype.categorical])
            numerical_column_size = len(col_names_dict[stype.numerical])
            categorical_stats_list = [
                col_stats[col_name]
                for col_name in col_names_dict[stype.categorical]
            ]
            self.cat_encoder = EmbeddingEncoder(out_channels=channels,
                                                stats_list=categorical_stats_list,
                                                stype=stype.categorical)
            self.num_norm = LayerNorm(numerical_column_size)
            self.tab_transformer_convs = ModuleList([
                TabTransformerConv(
                    channels=channels,
                    num_heads=num_heads,
                ) for _ in range(num_layers)
            ])
            self.decoder = Sequential(
                Linear(categorical_col_len * channels + numerical_column_size,
                    out_channels * 2), ReLU(),
                Linear(out_channels * 2, out_channels))

        def forward(self, tf: TensorFrame) -> Tensor:

            B, _ = tf.feat_dict[stype.categorical].shape
            feat_cat = self.cat_encoder(tf.feat_dict[stype.categorical])

            for tab_transformer_conv in self.tab_transformer_convs:
                feat_cat = tab_transformer_conv(feat_cat)

            feat_cat = feat_cat.reshape(B, -1)

            feat_num = self.num_norm(tf.feat_dict[stype.numerical])

            x = torch.cat((feat_cat, feat_num), dim=1)

            out = self.decoder(x)
            return out

In the constructor, you can specify the encoder, convolution and decoder.
In the example above, :class:`EmbeddingEncoder` is used to encode the categorical features.
The categorical embeddings are then passed into layers of :class:`TabTransformerConv`.
:class:`LayerNorm` is applied to numerical features.
Then the outputs are concatenated and fed into an MLP decoder.

Let's create train-test split and create data loaders.

.. code-block:: python

    from torch_frame.data import DataLoader

    train_dataset, test_dataset = dataset[:0.8], dataset[0.80:]
    train_loader = DataLoader(train_dataset.tensor_frame, batch_size=128,
                            shuffle=True)
    test_loader = DataLoader(test_dataset.tensor_frame, batch_size=128,
                            shuffle=True)

Let’s train this model on the training nodes for 50 epochs:

.. code-block:: python

    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabTransformer(
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

Finally, we can evaluate our model on the test split:

.. code-block:: python

    model.eval()
    pred = model(test_dataset.tensor_frame).argmax(dim=1)
    pred_class = pred.argmax(dim=-1)
    correct = float((tf.y == pred_class).sum())
    acc = int(correct) / len(tf.y)
    print(f'Accuracy: {acc:.4f}')
    >>> Accuracy: 0.8235

This is all it takes to implement your first deep tabular network.
Happy hacking!
