import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from torch_frame.nn.decoder import Decoder


class TromptDownstream(Decoder):
    """The Trompt downstream introduced in https://arxiv.org/abs/2305.18446

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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_prompts = num_prompts
        self.lin1 = Linear(in_channels, 1)
        self.lin2 = Linear(in_channels, in_channels)
        self.lin3 = Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x: Tensor):
        batch_size = len(x)
        assert x.shape == (batch_size, self.num_prompts, self.in_channels)
        # [batch_size, num_prompts, 1]
        w_prompt = F.softmax(self.lin1(x), dim=1)
        # [batch_size, in_channels]
        x = (w_prompt * x).sum(dim=1)
        # [batch_size, out_channels]
        x = self.lin3(F.relu(self.lin2(x)))
        return x
