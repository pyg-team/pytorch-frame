Modular Design of Deep Tabular Models
=====================================

Many recent strong tabular deep models follow modular design (Encoder, Table convolution, Decoder). We design the overall architecture of deep tabular models in :pyg:`PyTorch Frame` in the following image.

.. figure:: ../_figures/modular.png
  :align: center
  :width: 100%

Encoder
-------

Feature Encoder transforms input :obj:`TensorFrame` into :obj:`Tensor`. This class can contain learnable parameters and missing value handling. Each feature encoder contains
an `stype_encoder_dict` that contains key value pairs of :obj:`stype` to :obj:`StypeEncoder`.

:obj:`StypeEncoder` encodes :obj:`tensor` of a specific stype into 3-dimensional column-wise tensor that is input into :class:`TableConv`.

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
