import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module
from torch.nn.init import zeros_

from torch_frame.nn.conv import TableConv

from ..utils.init import attenuated_kaiming_uniform_


class TabTransformerConv(TableConv):
    def __init__(self):
        pass
