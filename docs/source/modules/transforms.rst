torch_frame.transforms
======================

.. contents:: Contents
    :local:

.. currentmodule:: torch_frame.transforms

Transforms
----------

:pyf:`PyTorch Frame` allows for data transformation across different :obj:`stype`'s or within the same :obj:`stype`. Transforms takes in both :class:`TensorFrame` and column stats.

Let's look an example, where we apply `CatToNumTransform <https://dl.acm.org/doi/10.1145/507533.507538>`_ to transform the categorical features into numerical features.

.. code-block:: python

    from torch_frame.datasets import Yandex
    from torch_frame.transforms import CatToNumTransform
    from torch_frame import stype
    from torch_frame.typing import TrainingStage

    dataset = Yandex(root='/tmp/adult', name='adult')
    dataset.materialize()
    transform = CatToNumTransform()
    train_dataset = dataset.get_split(TrainingStage.TRAIN)

    train_dataset.tensor_frame.col_names_dict[stype.categorical]
    >>> ['C_feature_0', 'C_feature_1', 'C_feature_2', 'C_feature_3', 'C_feature_4', 'C_feature_5', 'C_feature_6', 'C_feature_7']

    test_dataset = dataset.get_split(TrainingStage.TEST)
    transform.fit(train_dataset.tensor_frame, dataset.col_stats)

    transformed_col_stats = transform.transformed_stats

    transformed_col_stats.keys()
    >>> dict_keys(['C_feature_0_0', 'C_feature_1_0', 'C_feature_2_0', 'C_feature_3_0', 'C_feature_4_0', 'C_feature_5_0', 'C_feature_6_0', 'C_feature_7_0'])

    transformed_col_stats['C_feature_0_0']
    >>> {<StatType.MEAN: 'MEAN'>: 0.6984029484029484, <StatType.STD: 'STD'>: 0.45895127199411595, <StatType.QUANTILES: 'QUANTILES'>: [0.0, 0.0, 1.0, 1.0, 1.0]}

    transform(test_dataset.tensor_frame)
    >>> TensorFrame(
          num_cols=14,
          num_rows=16281,
          numerical (14): ['N_feature_0', 'N_feature_1', 'N_feature_2', 'N_feature_3', 'N_feature_4', 'N_feature_5', 'C_feature_0_0', 'C_feature_1_0', 'C_feature_2_0', 'C_feature_3_0', 'C_feature_4_0', 'C_feature_5_0', 'C_feature_6_0', 'C_feature_7_0'],
          has_target=True,
          device=cpu,
        )

You can see that after the transform, the column names of the categorical features changes and the categorical features are transformed into numerical features.


.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_frame.transforms.functions %}
     {{ name }}
   {% endfor %}
