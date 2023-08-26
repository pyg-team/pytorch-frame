from torch.nn import Module, Linear, PReLU
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_, zeros_
from torch_frame.nn.decoder import Decoder

class ExcelFormerPredictionHead(Decoder):
    def __init__(self, in_channels, out_channels, num_features):
        super().__init__()
        self.channels = in_channels
        self.C = out_channels
        self.W = Linear(num_features, self.C)
        self.W_d = Linear(self.channels, 1)

    def reset_parameters(self):
        xavier_uniform_(self.W.weight)
        zeros_(self.W.bias)
        xavier_uniform_(self.W_d.weight)
        zeros_(self.W_d.bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x @ self.W.weight + self.W.bias
        x = PReLU(x)
        x = self.W_d(x)
        return x.squeeze(1)