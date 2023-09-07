from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn.modules.module import Module
from xgboost import XGBClassifier

import torch_frame
from torch_frame import stype
from torch_frame.gbdt import GradientBoostingDecisionTree


class XGBoost(GradientBoostingDecisionTree):
    def __init__(self, task_type):
        pass

    def fit_tune(self):
        pass
