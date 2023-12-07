Handling Heterogeneous Stypes
=============================

:pyf:`PyTorch Frame` supports many heterogeneous :obj:`stypes<torch_frame.stype>`, including but not limited to :class:`~stype.multicategorical`, :class:`~stype.timestamp` and :class:`~stype.embedding`.
In this tutorial, we will show you a simple example of handling heterogeneous data types with :pyf:`PyTorch Frame`.

.. contents::
    :local:

Handling Heterogeneous Columns
------------------------------
First, let us create a sample dataset with many different stypes.

.. code-block:: none

    import random

    import numpy as np
    import pandas as pd

    # Numerical column
    numerical = np.random.randint(0, 100, size=10)

    # Categorical column
    simple_categories = ['Type 1', 'Type 2', 'Type 3']
    categorical = np.random.choice(simple_categories, size=100)

    # Timestamp column
    time = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Multicategorical column
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    multicategorical = [
        random.sample(categories, k=random.randint(0, len(categories)))
        for _ in range(100)
    ]

    # Embedding column (assuming an embedding size of 5 for simplicity)
    embedding_size = 5
    embedding = np.random.rand(100, embedding_size)

    # Create the DataFrame
    df = pd.DataFrame({
        'Numerical': numerical,
        'Categorical': categorical,
        'Time': time,
        'Multicategorical': multicategorical,
        'Embedding': list(embedding)
    })

    df.head()
    >>>
        Numerical Categorical       Time                                  Multicategorical                                          Embedding
    0         44      Type 2 2023-01-01              [Category D, Category A, Category B]  [0.2879910043632805, 0.38346222503494787, 0.74...
    1         47      Type 2 2023-01-02  [Category C, Category A, Category B, Category D]  [0.0923738894608982, 0.3540466620838102, 0.551...
    2         64      Type 2 2023-01-03                          [Category D, Category C]  [0.3209972413734975, 0.22126268518378278, 0.14...
    3         67      Type 1 2023-01-04                          [Category C, Category A]  [0.2603409275874047, 0.5370225213757797, 0.447...
    4         67      Type 2 2023-01-05                                      [Category A]  [0.46924917399024213, 0.8411401297855995, 0.90...


Now let's load the :class:`pandas.DataFrame` into :class:`torch_frame.data.Dataset` class so that we have :class:`~torch_frame.data.tensor_frame.TensorFrame` representation of the :class:`pandas.DataFrame`.

.. code-block:: none

    dataset = Dataset(
        df, col_to_stype={
            'Numerical': stype.numerical,
            'Categorical': stype.categorical,
            'Time': stype.timestamp,
            'Multicategorical': stype.multicategorical,
            'Embedding': stype.embedding
        })
    dataset.materialize()

    dataset.tensor_frame
    >>> TensorFrame(
        num_cols=4,
        num_rows=100,
        categorical (1): ['Categorical'],
        timestamp (1): ['Time'],
        multicategorical (1): ['Multicategorical'],
        embedding (1): ['Embedding'],
        has_target=True,
        device='cpu',
        )

For each :class:`~torch_frame.stype`, we need to specify its encoder in :obj:`stype_encoder_dict`.

.. code-block:: python

    from torch_frame.nn.encoder.stype_encoder import (
        EmbeddingEncoder,
        LinearEmbeddingEncoder,
        LinearEncoder,
        MultiCategoricalEmbeddingEncoder,
        TimestampEncoder,
    )
    from torch_frame.typing import NAStrategy

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.embedding: LinearEmbeddingEncoder(),
        stype.multicategorical: MultiCategoricalEmbeddingEncoder(),
        stype.timestamp: TimestampEncoder(na_strategy=NAStrategy.MEDIAN_TIMESTAMP)
    }

Now we can specify the :obj:`stype_encoder_dict` to a model of your choice.

.. note::
    Some pre-implemented models do not support all :obj:`stypes<torch_frame.stype>`.
    For example, :class:`~torch_frame.nn.models.TabTransformer` only supports numerical and categorical :obj:`stypes<torch_frame.stype>`.

.. code-block:: none

    from torch_frame.nn.models.ft_transformer import FTTransformer
    model = FTTransformer(
        channels=16,
        out_channels=1,
        num_layers=2,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    )

    model(dataset.tensor_frame)
    >>> tensor([[ 0.9405],
        [ 0.3857],
        [ 0.5265],
        [-0.3747],
        [ 0.7496],
        [ 0.0486],
        [ 0.2895],
        [ 0.1326],
        [ 0.4388],
        [-0.1665]], grad_fn=<AddmmBackward0>)

Auto Inference of Semantic Types
--------------------------------
We offer a simple utility function :class:`~torch_frame.utils.infer_df_stype` where you can automatically infer the :class:`~torch_frame.stype` of different columns in the provided :class:`~pandas.DataFrame`.

.. code-block:: none

    infer_df_stype(df)
    >>> {'Numerical': <stype.numerical: 'numerical'>,
        'Categorical': <stype.categorical: 'categorical'>,
        'Time': <stype.timestamp: 'timestamp'>,
        'Multicategorical': <stype.multicategorical: 'multicategorical'>,
        'Embedding': <stype.embedding: 'embedding'>}

However, the inference may not be always correct/best for your data.
We recommend you double-checking the correctness yourself before actually using it.


Dealing with Complex Raw Data
-----------------------------

Often times the raw data from a dataset can be complex.
For example, different multicategorical columns can have different delimiters, and different time columns can have different time formats.

Currently, raw column data of type :class:`list` or :class:`str` are supported for :class:`~torch_frame.stype.multicategorical`.
You can also specify different delimiters for different columns through :obj:`col_to_sep` argument in :class:`torch_frame.data.Dataset`.
If a string is specified, the same delimiter will be used throughout all the multicategorical columns.
If a dictionary is given, we use a different delimiter specified for each column.

.. note::
    You need to sepecify delimiters for all multicategorical columns where the raw data is :class:`str`, otherwise the value of each cell would be considered as one categorical value.

Here is an example of handing a :class:`~pandas.DataFrame` with multiple multicategorical columns.

.. code-block:: python

    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    multicategorical1 = [
        random.sample(categories, k=random.randint(0, len(categories)))
        for _ in range(100)
    ]
    multicategorical2 = [
        ','.join(random.sample(categories, k=random.randint(0, len(categories))))
        for _ in range(100)
    ]
    multicategorical3 = [
        '/'.join(random.sample(categories, k=random.randint(0, len(categories))))
        for _ in range(100)
    ]
    # Create the DataFrame
    df = pd.DataFrame({
        'Multicategorical1': multicategorical1,
        'Multicategorical2': multicategorical2,
        'Multicategorical3': multicategorical3,
    })

    dataset = Dataset(
        df, col_to_stype={
            'Multicategorical1': stype.multicategorical,
            'Multicategorical2': stype.multicategorical,
            'Multicategorical3': stype.multicategorical,
        }, col_to_sep={'Multicategorical2': ',', 'Multicategorical3': '/'})

    dataset.col_stats
    >>>> {'Multicategorical1': {<StatType.MULTI_COUNT: 'MULTI_COUNT'>:
    (['Category B', 'Category D', 'Category A', 'Category C'], [61, 60, 56, 49])},
    'Multicategorical2': {<StatType.MULTI_COUNT: 'MULTI_COUNT'>:
    (['Category D', 'Category A', 'Category B', 'Category C'], [53, 52, 51, 46])},
    'Multicategorical3': {<StatType.MULTI_COUNT: 'MULTI_COUNT'>:
    (['Category D', 'Category B', 'Category C', 'Category A'], [52, 52, 51, 46])}}

For :class:`~torch_frame.stype.timestamp`, you can similarly specify the time format in :obj:`col_to_time_format`.
See `strfttime documentation <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`_ for more information on formats.
If not specified, pandas's internal :class:`pandas.to_datetime` function will be used to auto parse time columns.

.. code-block:: none

    dates = pd.date_range(start="2023-01-01", periods=5, freq='D')

    df = pd.DataFrame({
            'Time1': dates,  # ISO 8601 format (default)
            'Time2': dates.strftime('%Y-%m-%d %H:%M:%S'),
    })

    df.head()
    >>>        Time1                Time2
        0 2023-01-01  2023-01-01 00:00:00
        1 2023-01-02  2023-01-02 00:00:00
        2 2023-01-03  2023-01-03 00:00:00
        3 2023-01-04  2023-01-04 00:00:00
        4 2023-01-05  2023-01-05 00:00:00

    dataset = Dataset(
        df, col_to_stype={
            'Time1': stype.timestamp,
            'Time2': stype.timestamp,
        }, col_to_time_format='%Y-%m-%d %H:%M:%S')

    dataset.materialize()

    dataset.col_stats
    >>> {'Time1': {<StatType.YEAR_RANGE: 'YEAR_RANGE'>: [2023, 2023],
    <StatType.NEWEST_TIME: 'NEWEST_TIME'>: tensor([2023,    0,    4,    3,    0,    0,    0]),
    <StatType.OLDEST_TIME: 'OLDEST_TIME'>: tensor([2023,    0,    0,    6,    0,    0,    0]),
    <StatType.MEDIAN_TIME: 'MEDIAN_TIME'>: tensor([2023,    0,    2,    1,    0,    0,    0])},
    'Time2': {<StatType.YEAR_RANGE: 'YEAR_RANGE'>: [2023, 2023],
    <StatType.NEWEST_TIME: 'NEWEST_TIME'>: tensor([2023,    0,    4,    3,    0,    0,    0]),
    <StatType.OLDEST_TIME: 'OLDEST_TIME'>: tensor([2023,    0,    0,    6,    0,    0,    0]),
    <StatType.MEDIAN_TIME: 'MEDIAN_TIME'>: tensor([2023,    0,    2,    1,    0,    0,    0])}}
