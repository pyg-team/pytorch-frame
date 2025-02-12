r"""Model package."""
from .trompt import Trompt
from .ft_transformer import FTTransformer
from .excelformer import ExcelFormer
from .tabnet import TabNet
from .resnet import ResNet
from .tab_transformer import TabTransformer
from .mlp import MLP
from .bcauss import BCAUSS
from .cfr import CFR

__all__ = classes = [
    'Trompt', 'FTTransformer', 'ExcelFormer', 'TabNet', 'ResNet',
    'TabTransformer', 'MLP', 'BCAUSS', 'CFR'
]
