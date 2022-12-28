import os
import pytest
from neural_network_pdes.euler import (
    euler_loss,
    interior_loss,
    left_dirichlet_bc_loss,
    right_dirichlet_bc_loss,
)
from neural_network_pdes.transform_network import fixed_rotation_layer
import torch


def test_poly_convolution_2d_produces_correct_sizes():

    assert True is True


def test_transform_network():

    ans = fixed_rotation_layer(2)
    res = torch.tensor(ans.weight)
    assert res.shape == torch.Size([4, 2])

    ans = fixed_rotation_layer(3)
    res = torch.tensor(ans.weight)
    assert res.shape == torch.Size([9, 3])
