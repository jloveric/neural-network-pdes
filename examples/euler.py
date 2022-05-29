from typing import List

from torchmetrics import Metric
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from pytorch_lightning import LightningModule, Trainer
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn import LazyBatchNorm1d, LayerNorm


def pde_grid():

    x = 2.0 * (torch.arange(0, 100, 1) / 100.0 - 0.5)
    t = torch.arange(0.0, 100, 1) / (100.0)
    grid_x, grid_t = torch.meshgrid(x, t)
    return torch.transpose(torch.stack([grid_x.flatten(), grid_t.flatten()]), 0, 1)


class PDEDataset(Dataset):
    def __init__(self, rotations: int = 1):

        # interior conditions
        x = 2.0 * torch.rand(10000) - 1.0
        t = torch.rand(10000)

        # boundary conditions
        xl = -torch.ones(1000)
        xr = torch.ones(1000)
        tb = torch.rand(1000)

        # Define the "grid" points in the model
        # These could all be generated on the fly
        # Add initial conditions which start at time t=0
        # Each x has a correspoding time, so below the intial
        # conditions are all the interior conditions (x,t)
        # then all conditions at t=0, (x,0) then the left
        # boundary conditions at various times (xl, tb) and
        # the right boundary conditions at various times (xr, tb)
        x = torch.cat((x, x, xl, xr)).view(-1, 1)
        t = torch.cat((t, 0 * t, tb, tb)).view(-1, 1)

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


def interior_loss(q: Tensor, grad_q: Tensor):
    """
    Compute the loss (solve the PDE) for everywhere not
    on a boundary or an initial condition.
    Args :
        q : primitive variables vector
        grad_q : gradient of the primitive variables [dx,dt]
    Returns :
        difference of pde from 0 squared
    """

    gamma = 1.4

    r = q[:, 0]
    u = q[:, 1]
    p = q[:, 2]

    rt = grad_q[:, 0, 1]
    rx = grad_q[:, 0, 0]

    ut = grad_q[:, 1, 1]
    ux = grad_q[:, 1, 0]

    pt = grad_q[:, 2, 1]
    px = grad_q[:, 2, 0]

    c2 = gamma * p / r
    # Note, the equations below are multiplied by r to reduce the loss.
    r_eq = torch.stack(
        [
            rt + u * rx + r * ux,
            r * (ut + u * ux + (1 / r) * px),
            (pt + r * c2 * ux + u * px),
        ]
    )

    # and then reduced by a factor 1000 to further shrink
    res = torch.dot(r_eq.flatten(), r_eq.flatten()) / 1
    return res


def euler_loss(x: Tensor, q: Tensor, grad_q: Tensor, targets: Tensor):
    """
    Compute the loss for the euler equations

    Args :
        x : netork inputs (positions and time).
        q : primitive variable vector.
        grad_q : gradients of primitive values [dx, dt]
        targets : target values (used for boundaries and initial conditions.)
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

    in_loss = interior_loss(q[interior], grad_q[interior])
    ic_loss = initial_condition_loss(q[ic_indexes], targets[ic_indexes])
    left_bc_loss = left_dirichlet_bc_loss(q[left_indexes], targets[left_indexes])
    right_bc_loss = right_dirichlet_bc_loss(q[right_indexes], targets[right_indexes])

    return in_loss, ic_loss, left_bc_loss, right_bc_loss


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # TODO: Add a fixed layer after the input layer that
        # produces [x, t, (x+t), (x-t)] with no weight adjustment
        # this would be a fixed linear transform layer.

        self.model = HighOrderMLP(
            layer_type=cfg.mlp.layer_type,
            n=cfg.mlp.n,
            n_in=cfg.mlp.n_in,
            n_hidden=cfg.mlp.n_in,
            n_out=cfg.mlp.n_out,
            in_width=cfg.mlp.input.width,
            in_segments=cfg.mlp.input.segments,
            out_width=cfg.mlp.output.width,
            out_segments=cfg.mlp.output.segments,
            hidden_width=cfg.mlp.hidden.width,
            hidden_layers=cfg.mlp.hidden.layers,
            hidden_segments=cfg.mlp.hidden.segments,
            normalization=None if cfg.mlp.normalize is False else LazyBatchNorm1d(),
        )
        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        self.train_dataset = PDEDataset(rotations=self.cfg.rotations)
        self.test_dataset = PDEDataset(rotations=self.cfg.rotations)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad_(True)
        y_hat = self(x)

        jacobian_list = []

        # We need to perform this operation per element instead of once per batch.  If you do it for a batch in computes
        # the gradient of all the batch inputs vs all the batch outputs (which is mostly zeros).  They need an operation
        # that computes the gradient for each input output pair.  Shouldn't be this slow.
        # This should use vmap
        for inp in x:
            inp = inp.unsqueeze(dim=0)
            jacobian = torch.autograd.functional.jacobian(
                self, inp, create_graph=True, strict=False, vectorize=True
            )
            jacobian_list.append(jacobian)

        gradients = torch.reshape(torch.stack(jacobian_list), (-1, 3, 2))

        in_loss, ic_loss, left_bc_loss, right_bc_loss = euler_loss(
            x=x, q=y_hat, grad_q=gradients, targets=y
        )

        loss = in_loss + ic_loss + left_bc_loss + right_bc_loss

        self.log(f"in_loss", in_loss)
        self.log(f"ic_loss", ic_loss)
        self.log(f"left_bc_loss", left_bc_loss)
        self.log(f"right_bc_loss", right_bc_loss)
        self.log(f"train_loss", loss, prog_bar=True)

        return loss

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=10,
        )
        return trainloader

    def test_dataloader(self):

        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=10,
        )
        return testloader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


@hydra.main(config_path="../config", config_name="euler")
def run_implicit_images(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch:03d}", monitor="train_loss"
        )
        trainer = Trainer(
            max_epochs=cfg.max_epochs, gpus=cfg.gpus, callbacks=[checkpoint_callback]
        )
        model = Net(cfg)
        trainer.fit(model)
        print("testing")
        trainer.test(model)

        print("finished testing")
        print("best check_point", trainer.checkpoint_callback.best_model_path)
    else:
        # plot some data
        print("evaluating result")
        print("cfg.checkpoint", cfg.checkpoint)

        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        this_path = f"{hydra.utils.get_original_cwd()}"
        print("checkpoint_path", checkpoint_path)

        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()
        inputs = pde_grid().detach()
        y_hat = model(inputs).detach().numpy()
        outputs = y_hat.reshape(100, 100, 3)

        names = ["Density", "Velocity", "Energy"]
        for j, name in enumerate(names):

            plt.figure(j + 1)
            fig, (ax0, ax1) = plt.subplots(2, 1)

            # The outputs are density, momentum and energy
            # so each of the components 0, 1, 2 represents
            # on of those quantities
            c = ax0.pcolor(outputs[:, :, j])
            ax0.set_xlabel("x")
            ax0.set_ylabel("time")

            for i in range(0, 100, 20):
                if name == "Velocity":
                    d = ax1.plot(outputs[:, i, j] / outputs[:, i, 0], label=f"t={i}")
                else:
                    d = ax1.plot(outputs[:, i, j], label=f"t={i}")

                ax1.set_xlabel("x")
                ax1.set_ylabel(f"{name}")

            ax1.legend()

            ax0.set_title(f"{name} with {cfg.mlp.layer_type} layers")
            plt.xlabel("x")

            if cfg.save_images is True:
                plt.savefig(
                    f"{this_path}/images/{name}-{cfg.mlp.layer_type}",
                    dpi="figure",
                    format=None,
                    metadata=None,
                    bbox_inches=None,
                    pad_inches=0.1,
                    facecolor="auto",
                    edgecolor="auto",
                    backend=None,
                )

        plt.show()


if __name__ == "__main__":
    run_implicit_images()
