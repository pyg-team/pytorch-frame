torch_frame.transforms
====================

.. contents:: Contents
    :local:

.. currentmodule:: torch_frame.transforms

Transforms
--------

:pyg:`PyTorch Frame` allows for data transformation across different :obj:`stype`'s or within the same :obj:`stype`. Transforms takes in both :obj:`TensorFrame` and column stats.

Let's look an example, where we apply `OrderedTargetStatisticsEncoder <https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-to-numberic>`_ to transform the categorical features into numerical features.

.. code-block:: python

    from torch_frame.datasets import Yandex
    from torch_frame.transforms import OrderedTargetStatisticsEncoder

    dataset = Yandex(root='/tmp/adult', name='adult')
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    transform = OrderedTargetStatisticsEncoder()
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


.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_frame.transforms.functions %}
     {{ name }}
   {% endfor %}
