from typing import List

from pytorch_lightning.metrics import Metric
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.metrics.functional import accuracy
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
# from high_order_mlp import HighOrderMLP
from torch.utils.data import DataLoader, Dataset


class PDEDataset(Dataset):
    def __init__(self, rotations: int = 1):

        # interior conditions
        x = (2.0*torch.rand(10000)-1.0).view(-1, 1)
        t = torch.rand(10000).view(-1, 1)

        # These could all be generated on the fly
        # Add initial conditions which start at time t=0
        x = torch.cat([x, x])
        t = torch.cat([t, 0*t])

        # initial condition
        rho = torch.where(x > 0, 1.0, 0.125)
        p = torch.where(x > 0, 1.0, 0.1)
        u = torch.where(x > 0, 0.0, 0.0)

        self.input = torch.cat([x, t], dim=1)
        self.output = torch.cat([rho, u, p], dim=1)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input[idx], self.output[idx]


def initial_condition_loss(outputs: Tensor, targets: Tensor):
    diff = targets-outputs
    return torch.dot(diff.flatten(), diff.flatten())


def left_dirichlet_bc_loss(outputs: Tensor, targets: Tensor):
    diff = targets-outputs
    return torch.dot(diff.flatten(), diff.flatten())


def right_dirichlet_bc_loss(outputs: Tensor, targets: Tensor):
    diff = targets-outputs
    return torch.dot(diff.flatten(), diff.flatten())


def interior_loss(q: Tensor, grad_q: Tensor):
    print('q.shape', q.shape)
    print('grad_q.shape', grad_q.shape)

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

    c2 = gamma*p/r

    r_eq = torch.stack([rt+u*rx+r*ux, ut+u*ux+(1/r)*px, pt+r*c2*ux+u*px])
    return torch.dot(r_eq.flatten(), r_eq.flatten())


def euler_loss(x: Tensor, q: Tensor, grad_q: Tensor, targets: Tensor):
    """
    Compute the loss for the euler equations

    Args :
        x : netork inputs (positions and time).
        q : primitive variable vector.
        grad_q : gradients of primitive values.
        targets : target values (used for boundaries and initial conditions.)
    """
    left_mask = (x[:, 0] == -1.0)  # x=0
    right_mask = (x[:, 0] == 1.0)  # x=1
    ic_mask = (x[:, 1] == 0)  # t=0

    left_indexes = torch.nonzero(left_mask)
    right_indexes = torch.nonzero(right_mask)
    ic_indexes = torch.nonzero(ic_mask)
    interior = torch.logical_not(left_mask+right_mask+ic_mask),

    loss = interior_loss(q[interior], grad_q[interior]) + \
        initial_condition_loss(q[ic_indexes], targets[ic_indexes]) + \
        left_dirichlet_bc_loss(q[left_indexes], targets[left_indexes]) + \
        right_dirichlet_bc_loss(q[right_indexes], targets[right_indexes])

    return loss


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
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
        )
        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):

        full_path = [f"{self.root_dir}/{path}" for path in self.cfg.images]
        # print('full_path', full_path)
        self.train_dataset = PDEDataset(
            rotations=self.cfg.rotations)
        self.test_dataset = PDEDataset(
            rotations=self.cfg.rotations)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad_(True)
        # print('x',x,'y',y)
        y_hat = self(x)

        # print('y_hat.grad', y_hat.grad)

        print('y_hat.shape', y_hat.shape)
        # deriv = torch.autograd.grad(outputs = [y_hat], inputs = [x], grad_outputs = torch.ones_like(y_hat) ,
        #                          allow_unused=True, retain_graph=True, create_graph=True)
        # deriv = torch.autograd.grad(outputs = [y_hat], inputs = [x], allow_unused=True, retain_graph=True, create_graph=True)
        # print('deriv', deriv)
        # print('y_hat', y_hat)
        jacobian_list = []

        # We need to perform this operation per element instead of once per batch.  If you do it for a batch in computes
        # the gradient of all the batch inputs vs all the batch outputs (which is mostly zeros).  They need an operation
        # that computes the gradient for each input output pair.  Shouldn't be this slow.
        for inp in x:
            inp = inp.unsqueeze(dim=0)
            jacobian = torch.autograd.functional.jacobian(
                self, inp, create_graph=False, strict=False, vectorize=False)
            jacobian_list.append(jacobian)

        gradients = torch.reshape(torch.stack(jacobian_list), (-1, 3, 2))
        # print('jcacobian', jacobian_list)
        #loss = self.loss(y_hat, y)
        loss = euler_loss(x=x, q=y_hat, grad_q=gradients, targets=y)

        self.log(f'train_loss', loss, prog_bar=True)
        # self.log(f'train_acc', acc, prog_bar=True)

        return loss

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=10)
        return trainloader

    def test_dataloader(self):

        testloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10)
        return testloader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


@ hydra.main(config_name="./config/euler")
def run_implicit_images(cfg: DictConfig):
    # TODO use a space filling curve to map x,y linear coordinates
    # to space filling coordinates 1d coordinate.
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
        model = Net(cfg)
        trainer.fit(model)
        print('testing')
        trainer.test(model)
        print('finished testing')
        print('best check_point', trainer.checkpoint_callback.best_model_path)
    else:
        # plot some data
        print('evaluating result')
        print('cfg.checkpoint', cfg.checkpoint)
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        print('checkpoint_path', checkpoint_path)
        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()
        image_dir = f"{hydra.utils.get_original_cwd()}/{cfg.images[0]}"
        output, inputs, image = image_to_dataset(
            image_dir, rotations=cfg.rotations)
        y_hat = model(inputs)
        max_x = torch.max(inputs, dim=0)
        max_y = torch.max(inputs, dim=1)
        print('x_max', max_x, 'y_max', max_y)
        print('y_hat.shape', y_hat.shape)
        print('image.shape', image.shape)
        ans = y_hat.reshape(image.shape[0], image.shape[1], image.shape[2])
        ans = (ans+1.0)/2.0
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(ans.detach().numpy())
        axarr[0].set_title('fit')
        axarr[1].imshow(image)
        axarr[1].set_title('original')
        plt.show()


if __name__ == "__main__":
    run_implicit_images()
