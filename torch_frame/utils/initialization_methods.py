import math

import torch
from torch import Tensor
from torch.nn.init import (
    _calculate_correct_fan,
    _calculate_fan_in_and_fan_out,
    _no_grad_uniform_,
    calculate_gain,
)


def attenuated_kaiming_uniform_(tensor: Tensor, scale: float = 0.1,
                                a: float = math.sqrt(5), mode: str = 'fan_in',
                                nonlinearity: str = 'leaky_relu'):
    r"""Attenuated Kaiming Uniform Initialization

    Args:
        x (tensor): Input tensor to be initialized
        scale (float): Positive rescaling constant to the variance.
        a (float): Negative slope of the rectifier used after this layer
        mode (str): Either 'fan_in' (default) or 'fan_out'. Choosing
        'fan_in' preserves the magnitude of the variance of the weights
        in the forward pass. Choosing 'fan_out' preserves the magnitudes
        in the backwards pass.
        nonlinearity (str) : the non-linear function (nn.functional name),
                    recommended to use only with 'relu' or 'leaky_relu'.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain * scale / math.sqrt(fan)
    bound = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def attenuated_xavier_uniform_(tensor: Tensor, scale: float = 0.1,
                               gain: float = 1.) -> Tensor:
    r"""Attenuated_xavier_uniform_

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * scale * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)
