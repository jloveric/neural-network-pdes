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


def by_left_eigenvector(rho, p, gamma, v0, v1, v2):
    """
    Multiply a vector - in this case the pde, by the
    left eigenvector assuming (continuity, velocity, pressure) equations
    in that order.  This is the primitive left eigenvector as derived
    in the jupyer notebook
    """
    c2 = torch.clamp(gamma * p / rho, 0.01)

    return (
        v0 - v2 / c2,
        v2 / (2 * c2) - rho * v1 / (2 * torch.sqrt(c2)),
        v2 / (2 * c2) + rho * v1 / (2 * torch.sqrt(c2)),
    )


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
    x: Tensor,
    q: Tensor,
    grad_q: Tensor,
    eps: float,
    time_decay: float = 0.0,
    scale_x: float = 1.0,
    scale_t: float = 1.0,
    solve_waves: bool=True,
):
    """
    Compute the loss (solve the PDE) for everywhere not
    on a boundary or an initial condition.
    Args :
        q : primitive variables vector
        grad_q : gradient of the primitive variables [dx,dt]
        eps : shock detection factor
        scale_x : inputs are scaled from [-1, 1] which compresses
        the derivatives x=x^{prime}*s where we take the derivative
        with respect to x so the scale factor is (1/s)
        scale_t : scale factor for t, see above
        solve_waves: If True, decompose with left eigenvalues into waves and
        solve those equations instead.
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

    delta = 1e-2

    continuity_equation = rt / scale_t + (u * rx + r * ux) / scale_x
    velocity_equation = ut / scale_t + (u * ux + (1 / r) * px) / scale_x
    pressure_equation = pt / scale_t + (r * c2 * ux + u * px) / scale_x
    if solve_waves is True :
        l1, l2, l3 = by_left_eigenvector(
            rho=r,
            p=p,
            gamma=gamma,
            v0=continuity_equation,
            v1=velocity_equation,
            v2=pressure_equation,
        )

        r_eq = torch.stack(
            [
                l1,
                l2,
                l3,
            ]
        )
    else :
        r_eq = torch.stack(
            [
                continuity_equation,
                velocity_equation,
                pressure_equation,
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
    scale_x: float = 1.0,
    scale_t: float = 1.0,
    solve_waves: bool=True,
):
    """
    Compute the loss for the euler equations

    Args :
        x : netork inputs (positions and time).
        q : primitive variable vector.
        grad_q : gradients of primitive values [dx, dt]
        targets : target values (used for boundaries and initial conditions.)
        eps : shock detection factor
        scale_x : inputs are scaled from [-1, 1] which compresses
        the derivatives x=x^{prime}*s where we take the derivative
        with respect to x so the scale factor is (1/s)
        scale_t : scale factor for t, see above
        solve_waves: If True, solve L*q instead of q.
    Returns :
        sum of losses.
    """
    left_mask = x[:, 0] == -1.0  # x=0
    right_mask = x[:, 0] == 1.0  # x=1
    ic_mask = x[:, 1] == -1.0  # t=-1

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
            x[interior],
            q[interior],
            grad_q[interior],
            eps=eps,
            time_decay=time_decay,
            scale_x=scale_x,
            scale_t=scale_t,
            solve_waves=solve_waves
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
