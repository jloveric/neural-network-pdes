from torch import Tensor
import torch
from high_order_layers_torch.layers import (
    high_order_fc_layers,
)
from high_order_layers_torch.networks import HighOrderMLP
from typing import Optional


def copy_layer():
    layer = torch.nn.Linear(2, 2, bias=False)
    weights = torch.tensor([[1, 0], [0, 1]])
    layer.weight = torch.nn.Parameter(weights, requires_grad=False)
    return layer


def fixed_rotation_layer(n: int):
    """
    Take n inputs and compute all the variations, n_i+n_j, n_i-n_j
    and create a layer that computes these with fixed weights. For
    n=2, outputs [x, t, 0.5*(x+t), 0.5*(x-t)]
    Args :
        - n: The number of inputs, would be 2 for (x, t)
    """

    combos = []
    for i in range(n):
        for j in range(i + 1, n):
            temp = [0] * n
            temp[j] += 0.5
            temp[i] += 0.5
            combos.append(temp)

            other = [0] * n
            other[i] = 0.5
            other[j] = -0.5
            combos.append(other)

    for i in range(n):
        temp = [0] * n
        temp[i] = 1
        combos.append(temp)

    layer = torch.nn.Linear(n, n*n, bias=False)
    weights = torch.tensor(combos)
    layer.weight = torch.nn.Parameter(weights, requires_grad=False)
    return layer


def rotation_transform_network(
    layer_type: str,
    in_width: int,
    hidden_width: int,
    out_width: int,
    n: int,
    n_in: int,
    n_hidden: int,
    n_out: int,
    in_segments: int,
    hidden_segments: int,
    out_segments: int,
    hidden_layers: int,
    scale: float,
    periodicity: float,
    normalize:bool=False
) -> torch.nn.Module:

    fixed_input = fixed_rotation_layer(n=in_width)

    mlp = HighOrderMLP(
        layer_type=layer_type,
        n=n,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        in_width=in_width*in_width,
        in_segments=in_segments,
        out_width=out_width,
        out_segments=out_segments,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        hidden_segments=hidden_segments,
        normalization=None if normalize is False else torch.nn.LazyBatchNorm1d,
        scale=scale,
        periodicity=periodicity
    )
    tl = [fixed_input, mlp]
    model = torch.nn.ModuleList(tl)
    return model
