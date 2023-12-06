import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, ReLU, Sequential

from torch_frame.nn.decoder import Decoder


class TromptDecoder(Decoder):
    r"""The Trompt downstream introduced in
    `"Trompt: Towards a Better Deep Neural Network for Tabular Data"
    <https://arxiv.org/abs/2305.18446>`_ paper.

    Args:
        in_channels (int): Input channel dimensionality
        out_channels (int): Output channel dimensionality
        num_prompts (int): Number of prompt columns.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_prompts: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_prompts = num_prompts
        self.lin_attn = Linear(in_channels, 1)
        self.mlp = Sequential(
            Linear(in_channels, in_channels),
            ReLU(),
            LayerNorm(in_channels),
            Linear(in_channels, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_attn.reset_parameters()
        for m in self.mlp:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = len(x)
        assert x.shape == (batch_size, self.num_prompts, self.in_channels)
        # [batch_size, num_prompts, 1]
        w_prompt = F.softmax(self.lin_attn(x), dim=1)
        # [batch_size, in_channels]
        x = (w_prompt * x).sum(dim=1)
        # [batch_size, out_channels]
        x = self.mlp(x)
        return x
