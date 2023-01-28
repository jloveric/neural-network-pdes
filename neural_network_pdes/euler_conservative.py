from typing import List

from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
import torch
from torch import Tensor
from torch.utils.data import Dataset
from neural_network_pdes.common import solution_points


class PDEDataset(Dataset):
    def __init__(self, size: int = 10000, gamma: float = 1.4):

        self._gamma = gamma

        x, t = solution_points(size)

        # initial condition
        rho = torch.where(x < 0, 1.0, 0.125)
        p = torch.where(x < 0, 1.0, 0.1)
        u = torch.where(x < 0, 0.0, 0.0)

        mv = rho * u
        energy = (p / (self._gamma - 1.0)) + 0.5 * mv * mv / rho

        self.input = torch.cat([x, t], dim=1)
        self.output = torch.cat([rho, mv, energy], dim=1)

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


def pressure(q: Tensor, gamma: float):
    rho = q[:, 0]
    mx = q[:, 1]
    en = q[:, 2]

    p = (gamma - 1.0) * (en - 0.5 * (mx * mx / rho))
    return p


def flux(q: Tensor, gamma: float = 1.4):

    p = pressure(q=q, gamma=gamma)
    rho = q[:, 0]
    mx = q[:, 1]
    en = q[:, 2]

    f = torch.zeros_like(q, requires_grad=True)
    with torch.no_grad():
        f[:, 0] = mx
        f[:, 1] = (mx * mx / rho) + p
        f[:, 2] = mx * (en + p) / rho
    return f


def interior_loss(
    x: Tensor,
    q: Tensor,
    grad_q: Tensor,
    grad_f: Tensor,
    hessian: Tensor = None,
    artificial_viscosity: float = 0,
    eps: float = 0.1,
    time_decay: float = 0.0,
    scale_x: float = 1.0,
    scale_t: float = 1.0,
):
    """
    Compute the loss (solve the PDE) for everywhere not
    on a boundary or an initial condition.
    Args :
        q : primitive variables vector
        grad_q : gradient of the primitive variables [dx,dt]
        hessian : second order derivatives with respect to input
        artificial_viscosity: add viscosity to ensure non entropy violating shocks
        eps : factor for discontinuity loss
    Returns :
        difference of pde from 0 squared
    """
    gamma = 1.4

    # this value should be 1 at t=0 and then move
    # towards 0 at t=end (which is 1 in this case)
    decay = 1.0 - time_decay * 0.5 * (x[:, 1] + 1)

    r = q[:, 0]
    mx = q[:, 1]
    e = q[:, 2]

    rt = grad_q[:, 0, 1]
    rx = grad_q[:, 0, 0]
    # rxx = hessian[:, 0, 0]

    mt = grad_q[:, 1, 1]
    grad_mx = grad_q[:, 1, 0]
    # mxx = hessian[:, 1, 0]

    et = grad_q[:, 2, 1]
    ex = grad_q[:, 2, 0]
    # exx = hessian[:, 1, 0]

    u = mx / r

    drhodx = rx
    dmdx = grad_mx
    dudx = (dmdx - u * drhodx) / r

    discontinuity = 1.0 / (eps * (torch.abs(dudx) - dudx) + 1)

    # frt = grad_f[:, 0, 1]
    frx = grad_f[:, 0, 0]

    # fmt = grad_f[:, 1, 1]
    fmx = grad_f[:, 1, 0]

    # fet = grad_f[:, 2, 1]
    fex = grad_f[:, 2, 0]

    # Note, the equations below are multiplied by r to reduce the loss.
    r_eq = torch.stack(
        [
            rt / scale_t + frx / scale_x,  # + artificial_viscosity * rxx,
            mt / scale_t + fmx / scale_x,  # + artificial_viscosity * mxx,
            et / scale_t + fex / scale_x,  # + artificial_viscosity * exx,
        ]
    )

    square = discontinuity * decay * r_eq * r_eq

    # and then reduced by a factor 1000 to further shrink
    res = torch.sum(square.flatten())
    return res


def euler_loss(
    x: Tensor,
    q: Tensor,
    grad_q: Tensor,
    grad_f: Tensor,
    hessian: Tensor,
    artificial_viscosity: float,
    targets: Tensor,
    eps: float = 0.1,
    time_decay: float = 0.0,
    scale_x: float = 1.0,
    scale_t: float = 1.0,
):
    """
    Compute the loss for the euler equations

    Args :
        x : netork inputs (positions and time).
        q : primitive variable vector.
        grad_q : gradients of primitive values [dx, dt]
        hessian : hessian with respect to inputs
        artificial_viscosity: add viscosity to prevent entropy violating shocks
        targets : target values (used for boundaries and initial conditions.)
        eps: factor for computing discontinuity loss
        time_decay: since things are causal late times should have smaller loss (I think)
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
            grad_f[interior],
            # hessian=hessian[interior],
            artificial_viscosity=artificial_viscosity,
            eps=eps,
            time_decay=time_decay,
            scale_x=scale_x,
            scale_t=scale_t,
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
