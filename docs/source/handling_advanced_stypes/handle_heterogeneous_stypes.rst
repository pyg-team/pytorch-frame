Handling Heterogeneous Stypes
=============================

:pyf:`PyTorch Frame` supports many heterogeneous stypes, including
:class:`stype.multicategorical`, :class:`stype.timestamp`, :class:`stype.embedding`.
In this tutorial, we will show how to deal with datasets with different stypes.

.. contents::
    :local:

Handling Heterogeneous Columns in a Benchmark Dataset
-----------------------------------------------------
First, let us create a dataset with many different stypes.

.. code-block:: none

    # Numerical column
    numerical = np.random.randint(0, 100,
                                size=100)

    # Categorical column
    simple_categories = ['Type 1', 'Type 2', 'Type 3']
    categorical = np.random.choice(simple_categories, size=100)

    # Time column
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


Now let's load the dataset

.. code-block:: none

    # Displaying the first few rows of the DataFrame
    print(df.head())
    >>>
        Numerical Categorical       Time                                  Multicategorical                                          Embedding
    0         44      Type 2 2023-01-01              [Category D, Category A, Category B]  [0.2879910043632805, 0.38346222503494787, 0.74...
    1         47      Type 2 2023-01-02  [Category C, Category A, Category B, Category D]  [0.0923738894608982, 0.3540466620838102, 0.551...
    2         64      Type 2 2023-01-03                          [Category D, Category C]  [0.3209972413734975, 0.22126268518378278, 0.14...
    3         67      Type 1 2023-01-04                          [Category C, Category A]  [0.2603409275874047, 0.5370225213757797, 0.447...
    4         67      Type 2 2023-01-05                                      [Category A]  [0.46924917399024213, 0.8411401297855995, 0.90...


    dataset = Dataset(
        df, col_to_stype={
            'Numerical': stype.numerical,
            'Categorical': stype.categorical,
            'Time': stype.timestamp,
            'Multicategorical': stype.multicategorical,
            'Embedding': stype.embedding
        })
    dataset.materialize()

    print(dataset.tensor_frame)
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

We can specify the :obj:`stype_encoder_dict` for each semantic types.

.. code-block:: none

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

Now we can run the model with a model.

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
    print(model(test_dataset.tensor_frame))

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
