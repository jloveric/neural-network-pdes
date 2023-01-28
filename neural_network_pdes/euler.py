from typing import List

from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
import torch
from torch import Tensor
from torch.utils.data import Dataset

from neural_network_pdes.common import solution_points


class PDEDataset(Dataset):
    def __init__(self, size: int = 10000):

        x, t = solution_points(size)

        # initial condition
        rho = torch.where(x < 0, 1.0, 0.125)
        p = torch.where(x < 0, 1.0, 0.1)
        u = torch.where(x < 0, 0.0, 0.0)

        self.input = torch.cat([x, t], dim=1)
        self.output = torch.cat([rho, u, p], dim=1)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input[idx], self.output[idx]


def initial_condition_loss(outputs: Tensor, targets: Tensor):
    diff = targets - outputs
    return torch.dot(diff.flatten(), diff.flatten())


def left_dirichlet_bc_loss(outputs: Tensor, targets: Tensor):
    diff = targets - outputs
    return torch.dot(diff.flatten(), diff.flatten())


def right_dirichlet_bc_loss(outputs: Tensor, targets: Tensor):
    diff = targets - outputs
    return torch.dot(diff.flatten(), diff.flatten())


def interior_loss(
    x: Tensor, q: Tensor, grad_q: Tensor, eps: float, time_decay: float = 0.0
):
    """
    Compute the loss (solve the PDE) for everywhere not
    on a boundary or an initial condition.
    Args :
        q : primitive variables vector
        grad_q : gradient of the primitive variables [dx,dt]
        eps : shock detection factor
    Returns :
        difference of pde from 0 squared
    """

    gamma = 1.4

    # assuming t ranges from 0 to 1
    decay = 1.0 - time_decay * x[:, 1]

    r = q[:, 0]
    u = q[:, 1]
    p = q[:, 2]

    rt = grad_q[:, 0, 1]
    rx = grad_q[:, 0, 0]

    ut = grad_q[:, 1, 1]
    ux = grad_q[:, 1, 0]

    pt = grad_q[:, 2, 1]
    px = grad_q[:, 2, 0]

    discontinuity = 1.0 / (eps * (torch.abs(ux) - ux) + 1)

    c2 = gamma * p / r
    # Note, the equations below are multiplied by r to reduce the loss.

    r_eq = torch.stack(
        [
            rt + u * rx + r * ux,
            # Normalize the value before
            # (r * ut + r * u * ux + px),
            (ut + u * ux + (1 / r) * px),
            (pt + r * c2 * ux + u * px),
        ]
    )

    square = discontinuity * decay * r_eq * r_eq

    res = torch.sum(square.flatten())

    return res


def euler_loss(
    x: Tensor,
    q: Tensor,
    grad_q: Tensor,
    targets: Tensor,
    eps: float,
    time_decay: float = 0.0,
):
    """
    Compute the loss for the euler equations

    Args :
        x : netork inputs (positions and time).
        q : primitive variable vector.
        grad_q : gradients of primitive values [dx, dt]
        targets : target values (used for boundaries and initial conditions.)
        eps : shock detection factor
    Returns :
        sum of losses.
    """
    left_mask = x[:, 0] == -1.0  # x=0
    right_mask = x[:, 0] == 1.0  # x=1
    ic_mask = x[:, 1] == 0  # t=0

    left_indexes = torch.nonzero(left_mask)
    right_indexes = torch.nonzero(right_mask)
    ic_indexes = torch.nonzero(ic_mask)
    interior = (torch.logical_not(left_mask + right_mask + ic_mask),)

    in_size = len(q[interior])
    ic_size = len(q[ic_indexes])
    lbc_size = len(q[left_indexes])
    rbc_size = len(q[right_indexes])

    in_loss = (
        interior_loss(
            x[interior], q[interior], grad_q[interior], eps=eps, time_decay=time_decay
        )
        / in_size
    )
    ic_loss = initial_condition_loss(q[ic_indexes], targets[ic_indexes]) / ic_size
    left_bc_loss = (
        left_dirichlet_bc_loss(q[left_indexes], targets[left_indexes]) / lbc_size
    )
    right_bc_loss = (
        right_dirichlet_bc_loss(q[right_indexes], targets[right_indexes]) / rbc_size
    )

    return in_loss, ic_loss, left_bc_loss, right_bc_loss
