from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from pytorch_lightning import LightningModule, Trainer, Callback
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import LazyBatchNorm1d, LazyInstanceNorm1d, LayerNorm
import io
import PIL.Image
from torchvision import transforms
from functorch import vmap, jacrev, hessian
from functorch.experimental import replace_all_batch_norm_modules_

from neural_network_pdes.euler import pde_grid, PDEDataset, euler_loss

import neural_network_pdes.euler as pform
import neural_network_pdes.euler_conservative as cform

from neural_network_pdes.transform_network import (
    transform_mlp,
    ReshapeNormalize,
    transform_low_mlp,
)
import logging

logger = logging.getLogger(__name__)


class SinLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


nonlinearity_options = {"relu": torch.nn.ReLU(), "sin": SinLayer(), "tanh": nn.Tanh()}


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.automatic_optimization = False

        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self._gamma = cfg.physics.gamma

        # TODO: Add a fixed layer after the input layer that
        # produces [x, t, (x+t), (x-t)] with no weight adjustment
        # this would be a fixed linear transform layer.

        nl = (
            None
            if cfg.mlp.nonlinearity is None
            else nonlinearity_options[cfg.mlp.nonlinearity]
        )
        if cfg.mlp.style == "standard":
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
                normalization=None if cfg.mlp.normalize is False else LazyBatchNorm1d,
                non_linearity=nl,
                periodicity=cfg.mlp.periodicity,
            )
        elif cfg.mlp.style == "transform":
            self.model = transform_mlp(
                layer_type=cfg.mlp.layer_type,
                in_width=cfg.mlp.input.width,
                hidden_width=cfg.mlp.hidden.width,
                out_width=cfg.mlp.output.width,
                n=cfg.mlp.n,
                n_in=cfg.mlp.n_in,
                n_out=cfg.mlp.n_out,
                n_hidden=cfg.mlp.n_hidden,
                in_segments=cfg.mlp.input.segments,
                hidden_segments=cfg.mlp.hidden.segments,
                out_segments=cfg.mlp.output.segments,
                hidden_layers=cfg.mlp.hidden.layers,
                normalization=None
                if cfg.mlp.normalize is False
                else ReshapeNormalize,  # LazyInstanceNorm1d,
                scale=cfg.mlp.scale,
                periodicity=cfg.mlp.periodicity,
            )
        elif cfg.mlp.style == "relu":
            self.model = LowOrderMLP(
                in_width=cfg.mlp.input.width,
                out_width=cfg.mlp.output.width,
                hidden_layers=cfg.mlp.hidden.layers,
                hidden_width=cfg.mlp.output.width,
                non_linearity=nl or nn.Tanh(),
                normalization=None,
            )
        elif cfg.mlp.style == "transform-relu":
            self.model = transform_low_mlp(
                in_width=cfg.mlp.input.width,
                out_width=cfg.mlp.output.width,
                hidden_layers=cfg.mlp.hidden.layers,
                hidden_width=cfg.mlp.output.width,
                non_linearity=nl or nn.Tanh(),
                normalization=None,
            )
        else:
            raise ValueError(
                f"Style should be 'standard' or 'transform', got {cfg.mlp.style}"
            )

        # replace_all_batch_norm_modules_(self.model)

        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def flux(self, x):
        return cform.flux(self.forward(x), gamma=self._gamma)

    def setup(self, stage: int):
        if self.cfg.form == "conservative":
            self.train_dataset = cform.PDEDataset(
                size=self.cfg.data_size, rotations=self.cfg.rotations
            )
            self.test_dataset = cform.PDEDataset(
                size=self.cfg.data_size, rotations=self.cfg.rotations
            )
        else:
            self.train_dataset = pform.PDEDataset(
                size=self.cfg.data_size, rotations=self.cfg.rotations
            )
            self.test_dataset = pform.PDEDataset(
                size=self.cfg.data_size, rotations=self.cfg.rotations
            )

    def training_step(self, batch: Tensor, batch_idx: int):
        optimizer = self.optimizers()

        x, y = batch

        x.requires_grad_(True)
        y_hat = self(x)

        xf = x.reshape(x.shape[0], 1, x.shape[1])
        jacobian = vmap(jacrev(self.forward))(xf)
        nj = jacobian.reshape(-1, 3, 2)

        if self.cfg.form == "conservative":
            flux_jacobian = vmap(jacrev(self.flux))(xf).reshape(-1, 3, 2)

            xl = xf.clone()
            xr = xf.clone()
            xl[:, 0] = xf[:, 0] - 0.5 * self.cfg.delta
            xr[:, 0] = xf[:, 0] + 0.5 * self.cfg.delta
            flux_left = vmap(jacrev(self.forward))(xl).reshape(-1, 3, 2)
            flux_right = vmap(jacrev(self.forward))(xr).reshape(-1, 3, 2)
            hess = (flux_right - flux_left) / self.cfg.delta
            # print('hessian', torch.nonzero(hess))
            # hess = vmap(hessian(self.forward))(xf).reshape(-1, 3, 4)

            in_loss, ic_loss, left_bc_loss, right_bc_loss = cform.euler_loss(
                x=x,
                q=y_hat,
                grad_q=nj,
                grad_f=flux_jacobian,
                hessian=hess,
                artificial_viscosity=self.cfg.physics.artificial_viscosity,
                targets=y,
                eps=self.cfg.loss_weight.discontinuity,
            )
        elif self.cfg.form == "integral":

            xl = xf.clone()
            xr = xf.clone()

            # dx
            xl[:, 0] = xf[:, 0] - 0.5 * self.cfg.delta
            xr[:, 0] = xf[:, 0] + 0.5 * self.cfg.delta

            # dt
            xtp = xf.clone()
            xtm = xf.clone()
            # xtp[:,1]=xf[:,1]+0.5*self.cfg.delta_t
            # xtm[:,1]=xf[:,1]-0.5*self.cfg.delta_t

            flux_left = vmap(self.flux)(xl).squeeze(1).unsqueeze(2)
            flux_right = vmap(self.flux)(xr).squeeze(1).unsqueeze(2)

            dudx = nj[:, 1, 0]
            grad_f = (flux_right - flux_left) / self.cfg.delta
            # print('hessian', torch.nonzero(hess))
            hess = vmap(hessian(self.forward))(xf).reshape(-1, 3, 4)

            in_loss, ic_loss, left_bc_loss, right_bc_loss = cform.euler_loss(
                x=x,
                q=y_hat,
                grad_q=nj,
                grad_f=grad_f,
                hessian=hess,
                artificial_viscosity=0,  # self.cfg.physics.artificial_viscosity,
                targets=y,
            )
        elif self.cfg.form == "primitive":

            in_loss, ic_loss, left_bc_loss, right_bc_loss = pform.euler_loss(
                x=x, q=y_hat, grad_q=nj, targets=y
            )
        else:
            raise ValueError(f"form should be conservative or primitive")

        loss = (
            self.cfg.loss_weight.interior * in_loss
            + self.cfg.loss_weight.initial * ic_loss
            + self.cfg.loss_weight.boundary * (left_bc_loss + right_bc_loss)
        )

        self.log(f"in_loss", in_loss)
        self.log(f"ic_loss", ic_loss)
        self.log(f"left_bc_loss", left_bc_loss)
        self.log(f"right_bc_loss", right_bc_loss)
        self.log(f"train_loss", loss, prog_bar=True)

        self.manual_backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        # If the network is discontinuous, add smoothing.
        smooth_discontinuous_network(self, factor=self.cfg.factor)

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["train_loss"])

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=10,
            drop_last=True,
        )
        return trainloader

    def test_dataloader(self):

        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=True,
        )
        return testloader

    def configure_optimizers(self):

        if self.cfg.optimizer.name == "adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.cfg.optimizer.lr,
            )
        elif self.cfg.optimizer.name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                max_iter=self.cfg.optimizer.max_iter,
            )

        reduce_on_plateau = False
        if self.cfg.optimizer.scheduler == "plateau":
            logger.info("Reducing lr on plateau")
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.cfg.optimizer.patience,
                factor=self.cfg.optimizer.factor,
                verbose=True,
            )
            reduce_on_plateau = True
        elif self.cfg.optimizer.scheduler == "exponential":
            logger.info("Reducing lr exponentially")
            lr_scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.cfg.optimizer.gamma
            )
        else:
            return optimizer

        scheduler = {
            "scheduler": lr_scheduler,
            "reduce_on_plateau": reduce_on_plateau,
            "monitor": "train_loss",
        }
        return [optimizer], [scheduler]


class ImageSampler(Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):

        images = generate_images(
            pl_module, save_to="memory", layer_type=self.cfg.mlp.layer_type
        )

        for index, image in enumerate(images):
            trainer.logger.experiment.add_image(
                f"img{index}", image, global_step=trainer.global_step
            )


def generate_images(model: nn.Module, save_to: str = None, layer_type: str = None):

    model.eval()
    inputs = pde_grid().detach().to(model.device)
    y_hat = model(inputs).detach().cpu().numpy()
    outputs = y_hat.reshape(100, 100, 3)

    names = ["Density", "Velocity", "Pressure"]

    image_list = []
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
            d = ax1.plot(outputs[:, i, j], label=f"t={i}")

            ax1.set_xlabel("x")
            ax1.set_ylabel(f"{name}")

        ax1.legend()

        ax0.set_title(f"{name} with {layer_type} layers")
        plt.xlabel("x")

        if save_to == "file":
            this_path = f"{hydra.utils.get_original_cwd()}"
            plt.savefig(
                f"{this_path}/images/{name}-{layer_type}",
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
            )
        elif save_to == "memory":
            buf = io.BytesIO()
            plt.savefig(
                buf,
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
            )
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image)
            image_list.append(image)
        plt.close()

    if save_to != "memory":
        plt.show()
    else:
        plt.close()
        return image_list

    plt.close()

    return None
