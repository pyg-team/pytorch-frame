import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

from torch_frame.nn.decoder import MLPDecoder
from torch_frame.nn.models import MLP


class EpsilonLayer(Module):
    def __init__(self):
        super().__init__()
        self.epsilon = Parameter(torch.randn(1, 1))

    def reset_parameters(self):
        self.epsilon.reset_parameters()

    def forward(self, t):
        return F.sigmoid(self.epsilon * torch.ones_like(t)[:, 0:1])


class BCAUSS(Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.epsilon = EpsilonLayer()
        # decoder for treatment group
        self.treatment_decoder = MLPDecoder()
        # decoder for control group
        self.control_decoder = MLPDecoder()

    def forward(self, x, t):
        r"""T stands for treatment and y stands for output."""
        out = self.mlp(x)
        if t == 0:
            out = self.control_decoder(out)
        else:
            out = self.treatment_decoder(out)
        penalty = self.epsilon(out)
        return out + penalty
