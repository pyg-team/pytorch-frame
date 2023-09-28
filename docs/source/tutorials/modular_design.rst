Modular Design of Deep Tabular Models
=====================================
Many recent strong tabular deep models follow modular design (Encoder, Table convolution, Decoder). We design the overall architecture of deep tabular models in :pyg:`PyTorch Frame` in the following image.

.. figure:: ../_figures/modular.png
  :align: center
  :width: 100%


Encoder
-------

:obj:`FeatureEncoder` transforms input :obj:`TensorFrame` into :obj:`Tensor`. This class can contain learnable parameters and missing value handling.

:obj:`StypeWiseFeatureEncoder` inherits from :obj:`FeatureEncoder`. It will take :obj:`TensorFrame` as input and apply stype-specific feature encoder (specified via `stype_encoder_dict`) to PyTorch :obj:`Tensor` of each stype to get embeddings for each stype.

The embeddings of different stypes are then concatenated along the column axis. In all, it transforms :obj:`TensorFrame` into 3-dimensional tensor `x` of shape [batch_size, num_cols, channels].
:obj:`StypeEncoder` encodes :obj:`tensor` of a specific stype into 3-dimensional column-wise tensor that is input into :class:`TableConv`. We have already implemented many encoders:

- :obj:`EmbeddingEncoder` is a :obj:`torch.nn.Embedding`-based encoder for categorical features
- :obj:`LinearBucketEncoder` is a bucket-based encoder for numerical features introduced in https://arxiv.org/abs/2203.05556

for a full list of :obj:`StypeEncoder`'s, you can take a look at :obj:`/torch_frame/encoder/stype_encoder.py`.

.. code-block:: python

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

The table convolution layer inherits from :obj:`TableConv`.

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

Decoder transforms the input column-wise :obj:`Tensor` into output tensor on which prediction head is applied.

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
