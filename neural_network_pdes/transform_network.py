from torch import Tensor
import torch
from high_order_layers_torch.layers import (
    high_order_fc_layers,
    fixed_rotation_layer as _fixed_rotation_layer
)
from high_order_layers_torch.networks import (
    transform_low_mlp,
    transform_mlp
)
from high_order_layers_torch.networks import HighOrderMLP, LowOrderMLP
from typing import Optional
import math


def fixed_rotation_layer(*args, **kwargs):
    layer = _fixed_rotation_layer(*args, **kwargs)
    if isinstance(layer, tuple):
        return layer[0]
    return layer

class ReshapeNormalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LazyInstanceNorm1d(track_running_stats=False)

    def __call__(self, x):
        y = x.unsqueeze(1)
        ans = self.norm(y).squeeze(1)
        return ans