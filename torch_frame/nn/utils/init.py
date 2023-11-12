import math

import torch
from torch import Tensor
from torch.nn.init import _calculate_correct_fan, calculate_gain


def attenuated_kaiming_uniform_(
    tensor: Tensor,
    scale: float = 0.1,
    a: float = math.sqrt(5),
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu',
) -> Tensor:
    r"""Attenuated Kaiming Uniform Initialization.

    Args:
        tensor (tensor): Input tensor to be initialized
        scale (float): Positive rescaling constant to the variance.
        a (float): Negative slope of the rectifier used after this layer
        mode (str): Either 'fan_in' (default) or 'fan_out'. Choosing
        'fan_in' preserves the magnitude of the variance of the weights
        in the forward pass. Choosing 'fan_out' preserves the magnitudes
        in the backwards pass.
        nonlinearity (str) : the non-linear function (nn.functional name),
                    recommended to use only with 'relu' or 'leaky_relu'.
    """
    with torch.no_grad():
        fan = _calculate_correct_fan(tensor, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain * scale / math.sqrt(fan)
        # Calculate uniform bounds from standard deviation
        bound = math.sqrt(3.0) * std
        return tensor.uniform_(-bound, bound)
