import torch


def pde_grid():

    x = 2.0 * (torch.arange(0, 100, 1) / 100.0 - 0.5)
    t = 2.0 * (torch.arange(0, 100, 1) / 100.0 - 0.5)
    grid_x, grid_t = torch.meshgrid(x, t)
    return torch.transpose(torch.stack([grid_x.flatten(), grid_t.flatten()]), 0, 1)


def solution_points(size: int):

    # interior conditions
    x = 2.0 * torch.rand(size) - 1.0
    t = 2.0 * torch.rand(size) - 1.0

    # boundary conditions (perhaps these should be sqrt interior...)
    boundary_size = int(size / 10)
    xl = -torch.ones(boundary_size)
    xr = torch.ones(boundary_size)
    tb = torch.rand(boundary_size)

    # initial conditions
    xic = 2.0 * torch.rand(boundary_size) - 1.0
    tic = torch.zeros(boundary_size) - 1.0

    # Define the "grid" points in the model
    # These could all be generated on the fly
    # Add initial conditions which start at time t=-1
    # Each x has a correspoding time, so below the intial
    # conditions are all the interior conditions (x,t)
    # then all conditions at t=-1, (x,-1) then the left
    # boundary conditions at various times (xl, tb) and
    # the right boundary conditions at various times (xr, tb)
    x = torch.cat((x, xic, xl, xr)).view(-1, 1)
    t = torch.cat((t, tic, tb, tb)).view(-1, 1)

    return x, t
