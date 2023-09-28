Modular Design of Deep Tabular Models
=====================================
Many recent strong tabular deep models follow modular design (Encoder, Table convolution, Decoder).
The overall design of deep tabular models in :pyg:`PyTorch Frame` follows the architecture show in the image below.

.. figure:: ../_figures/modular.png
  :align: center
  :width: 100%


The above image explains the high-level architecture of deep tabular models in :pyg:`PyTorchFrame`:

- First, the input dataframe is first converted to :class:`TensorFrame`, where data of each semantic type is stored separately.
- Then, the :obj:`TensorFrame` representing the dataset is fed into the Feature Encoder which converts it into three dimmensional :obj:`Tensor`'s.
- The :obj:`Tensor`'s are then concatenated into a single :obj:`Tensor` of shape [`batch_size`, `num_cols`, `num_channels`] and fed into layers of :class:`TableConv`.
- Finally, the output :obj:`Tensor` from the convolution is inputed into the decoder to produce the output :obj:`Tensor` of shape [`batch_size`, `out_channels`].

Encoder
-------

:obj:`FeatureEncoder` transforms input :obj:`TensorFrame` into :obj:`Tensor`. This class can contain learnable parameters and missing value handling.

:obj:`StypeWiseFeatureEncoder` inherits from :obj:`FeatureEncoder`. It takes :obj:`TensorFrame` as input and apply stype-specific feature encoder (specified via `stype_encoder_dict`) to PyTorch :obj:`Tensor` of each stype to get embeddings for each stype.

The embeddings of different stypes are then concatenated along the column axis.
In all, it transforms :obj:`TensorFrame` into 3-dimensional tensor `x` of shape [batch_size, num_cols, channels].

:obj:`StypeEncoder` encodes :obj:`tensor` of a specific stype into 3-dimensional column-wise tensor that is input into :class:`TableConv`. We have already implemented many encoders:

- :obj:`EmbeddingEncoder` is a :obj:`torch.nn.Embedding`-based encoder for categorical features
- :obj:`LinearBucketEncoder` is a bucket-based encoder for numerical features introduced in https://arxiv.org/abs/2203.05556
- :obj:`LinearPeriodicEncoder` utilizes sinusoidal functions to transform the input :obj:`Tensor` into a 3-dimensional tensor.
The encoding is defined using trainable parameters and includes the application of `sine`` and `cosine`` functions. The original encoding is described in https://arxiv.org/abs/2203.05556

For a full list of :obj:`StypeEncoder`'s, you can take a look at :obj:`/torch_frame/encoder/stype_encoder.py`.

`NaN` handling is accomplished in the :class:`StypeEncoder`.
By default, :class:`StypeEncoder` converts `NaN` values in each categorical feature to a new category and keeps the `NaN` values in numerical features.
With :class:`NAStrategy` specified, you can encode `NaN` values with specific `NaStrategy`.

A post module may also be supplied to an :obj:`StypeEncoder`.

A simple example is as follows:

.. code-block:: python

    from torch.nn import ReLU
    from torch_frame import NAStrategy
    from torch_frame.nn import EmbeddingEncoder

    encoder = EmbeddingEncoder(out_channels=8,
                                stats_list=stats_list,
                                stype=stype.categorical,
                                na_strategy=NAStrategy.MOST_FREQUENT,
                                post_module=ReLU())

:obj:`NaStrategy.MOST_FREQUENT` converts `NaN` values to the most frequent category.
Similarly, :obj:`NAStrategy.MEAN` :obj:`NAStrategy.ZEROS` is provided for numerical features.

Here is a example to create a Feature Encoder.
For each :obj:`Stype` in your :obj:`TensorFrame`, you'd need to specify a specific :obj:`StypeEncoder`.

.. code-block:: python

    from torch import LayerNorm
    from torch_frame import stype
    from torch_frame.nn import (
        EmbeddingEncoder,
        LinearBucketEncoder,
    )

    stype_encoder_dict = {
        stype.categorical:
        EmbeddingEncoder(),
        stype.numerical:
        LinearBucketEncoder(post_module=LayerNorm(channels)),
    }

    encoder = StypeWiseFeatureEncoder(
        out_channels=channels,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    )

Implementation of Convolution Layer
-----------------------------------

The table convolution layer inherits from :class:`TableConv`.
Table Convolution handles cross column interactions.

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from torch.nn import Linear
    from torch_frame.nn import TableConv

    class SelfAttentionConv(TableConv):
      r"""Simple self-attention-based table covolution to model interaction
      between different columns.

      Args:
          channels (int): Hidden channel dimensionality
      """
      def __init__(self, channels: int):
          super().__init__()
          self.channels = channels
          # Linear functions for modeling key/query/value in self-attention.
          self.lin_k = Linear(channels, channels)
          self.lin_q = Linear(channels, channels)
          self.lin_v = Linear(channels, channels)

      def forward(self, x: Tensor) -> Tensor:
          r"""Convolves input tensor to model interaction between different cols.

          Args:
              x (Tensor): Input tensor of shape [batch_size, num_cols, channels]

          Returns:
              out (Tensor): Output tensor of shape
                  [batch_size, num_cols, channels]
          """
          # [batch_size, num_cols, channels]
          x_key = self.lin_k(x)
          x_query = self.lin_q(x)
          x_value = self.lin_v(x)
          # [batch_size, num_cols, num_cols]
          prod = x_query.bmm(x_key.transpose(2, 1)) / math.sqrt(self.channels)
          # Attention weights between all pairs of columns.
          # Shape: [batch_size, num_cols, num_cols]
          attn = F.softmax(prod, dim=-1)
          # Mix `x_value` based on the attention weights
          # Shape: [batch_size, num_cols, num_channels]
          out = attn.bmm(x_value)
          return out

Initializing and calling it is straightforward.

.. code-block:: python

    conv = SelfAttentionConv(32)
    x = conv(x)


Decoder
-------

Decoder transforms the input column-wise :obj:`Tensor` into output :obj:`Tensor` on which prediction head is applied.
Here is an example implementation of a decoder:

.. code-block:: python

    import torch
    from torch import Tensor
    from torch.nn import Linear
    from torch_frame.nn import Decoder

    class MeanDecoder(Decoder):
        r"""Simple decoder that mean-pools over the embeddings of all columns and
        apply a linear transformation to map the pooled embeddings to desired
        dimensionality.

        Args:
            in_channels (int): Input channel dimensionality
            out_channels (int): Output channel dimensionality
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            # Linear function to map pooled embeddings into desired dimensionality
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x: Tensor) -> Tensor:
            # Mean pooling over the column dimension
            # [batch_size, num_cols, in_channels] -> [batch_size, in_channels]
            out = torch.mean(x, dim=1)
            # [batch_size, out_channels]
            return self.lin(out)
