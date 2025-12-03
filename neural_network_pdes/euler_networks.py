from typing import List
import torch_optimizer as alt_optim

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from Sophia import SophiaG
from lion_pytorch import Lion
from pytorch_lightning import LightningModule, Trainer, Callback
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import LazyBatchNorm1d, LazyInstanceNorm1d, LayerNorm
import matplotlib
import io
import PIL.Image
from torchvision import transforms
from torch import vmap
from torch.func import jacrev
from functorch.experimental import replace_all_batch_norm_modules_
from high_order_layers_torch.networks import (
    transform_low_mlp,
    transform_mlp,
    initialize_network_polynomial_layers,
)
import neural_network_pdes.euler as pform
import neural_network_pdes.euler_conservative as cform
from neural_network_pdes.common import pde_grid

from neural_network_pdes.transform_network import (
    ReshapeNormalize,
)
import logging

logger = logging.getLogger(__name__)
matplotlib.use("Agg")


class SinLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


nonlinearity_options = {"relu": torch.nn.ReLU(), "sin": SinLayer(), "tanh": nn.Tanh()}
normalization_type = {
    "midrange": MaxCenterNormalization,
    "maxabs": MaxAbsNormalization,
    "instance": LazyInstanceNorm1d,
}


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        device = cfg.accelerator

        self.automatic_optimization = False

        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self._gamma = cfg.physics.gamma

        nl = (
            None
            if cfg.mlp.nonlinearity is None
            else nonlinearity_options[cfg.mlp.nonlinearity]
        )
        if cfg.mlp.style == "transform":
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
                else normalization_type[
                    cfg.mlp.normalize
                ],  # MaxCenterNormalization, #MaxAbsNormalization,  # LazyInstanceNorm1d,
                scale=cfg.mlp.scale,
                periodicity=cfg.mlp.periodicity,
                rotations=cfg.mlp.rotations,
                resnet=cfg.mlp.resnet,
                device=device,
            )
        elif cfg.mlp.style == "high-order-input":
            layer_list = []
            input_layer = high_order_fc_layers(
                layer_type=cfg.mlp.layer_type,
                n=cfg.mlp.n,
                in_features=cfg.mlp.input.width,
                out_features=cfg.mlp.hidden.width,
                segments=cfg.mlp.input.segments,
                device=device,
            )
            layer_list.append(input_layer)

            # if normalization is not None:
            #    layer_list.append(normalization())

            lower_layers = LowOrderMLP(
                in_width=cfg.mlp.hidden.width,
                out_width=cfg.mlp.output.width,
                hidden_width=cfg.mlp.hidden.width,
                hidden_layers=cfg.mlp.hidden.layers - 1,
                non_linearity=nl,
                normalization=None
                if cfg.mlp.normalize is False
                else LazyInstanceNorm1d,
            )
            layer_list.append(lower_layers)

            self.model = nn.Sequential(*layer_list)
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

        initialize_network_polynomial_layers(self, max_slope=1.0, max_offset=0.0)
        # replace_all_batch_norm_modules_(self.model)

        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = nn.MSELoss()

    def forward(self, x):
        res = self.model(x)

        # limit density and pressure
        torch.clamp(res[:, 0], min=0.01)
        torch.clamp(res[:, 2], min=0.01)

        return res

    def flux(self, x):
        return cform.flux(self.forward(x), gamma=self._gamma)

    def setup(self, stage: int):
        if self.cfg.form == "conservative":
            self.train_dataset = cform.PDEDataset(size=self.cfg.data_size)
            self.test_dataset = cform.PDEDataset(size=self.cfg.data_size)
        else:
            self.train_dataset = pform.PDEDataset(size=self.cfg.data_size)
            self.test_dataset = pform.PDEDataset(size=self.cfg.data_size)

    def training_step(self, batch: Tensor, batch_idx: int):
        optimizer = self.optimizers()

        x, y = batch

        x.requires_grad_(True)
        # y.requires_grad_(True)
        y_hat = self(x)
        # y_hat.requires_grad_(True)

        xf = x.reshape(x.shape[0], 1, x.shape[1])
        jacobian = vmap(jacrev(self.forward))(xf)
        nj = jacobian.reshape(-1, 3, 2)

        if self.cfg.form == "conservative":
            flux_jacobian = vmap(jacrev(self.flux))(xf).reshape(-1, 3, 2)

            # xl = xf.clone()
            # xr = xf.clone()
            # xl[:, 0] = xf[:, 0] - 0.5 * self.cfg.delta
            # xr[:, 0] = xf[:, 0] + 0.5 * self.cfg.delta
            # flux_left = vmap(jacrev(self.forward))(xl).reshape(-1, 3, 2)
            # flux_right = vmap(jacrev(self.forward))(xr).reshape(-1, 3, 2)
            # hess = (flux_right - flux_left) / self.cfg.delta
            # print('hessian', torch.nonzero(hess))
            # hess = vmap(hessian(self.forward))(xf).reshape(-1, 3, 4)

            in_loss, ic_loss, left_bc_loss, right_bc_loss = cform.euler_loss(
                x=x,
                q=y_hat,
                grad_q=nj,
                grad_f=flux_jacobian,
                hessian=None,  # hess,
                artificial_viscosity=self.cfg.physics.artificial_viscosity,
                targets=y,
                eps=self.cfg.loss_weight.discontinuity,
                scale_x=self.cfg.scale_x,
                scale_t=self.cfg.scale_t,
            )
        elif self.cfg.form == "primitive":
            in_loss, ic_loss, left_bc_loss, right_bc_loss = pform.euler_loss(
                x=x,
                q=y_hat,
                grad_q=nj,
                targets=y,
                eps=self.cfg.loss_weight.discontinuity,
                time_decay=self.cfg.time_decay,
                scale_x=self.cfg.scale_x,
                scale_t=self.cfg.scale_t,
                solve_waves=self.cfg.solve_waves,
            )
        else:
            raise ValueError(f"form should be conservative or primitive")

        # in_loss.requires_grad_(True)
        # ic_loss.requires_grad_(True)
        # left_bc_loss.requires_grad_(True)
        # right_bc_loss.requires_grad_(True)
        loss = (
            self.cfg.loss_weight.interior * in_loss
            + self.cfg.loss_weight.initial * ic_loss
            + self.cfg.loss_weight.boundary * (left_bc_loss + right_bc_loss)
        )
        # loss.requires_grad_(True)

        self.log(f"in_loss", in_loss.item())
        self.log(f"ic_loss", ic_loss.item())
        self.log(f"left_bc_loss", left_bc_loss.item())
        self.log(f"right_bc_loss", right_bc_loss.item())
        self.log(f"train_loss", loss.item(), prog_bar=True)

        self.manual_backward(loss, create_graph=False)
        """
        if self.create_graph is False:
            self.manual_backward(loss, create_graph=self.create_graph)
        else:
            torch.autograd.grad(outputs=loss, inputs=x, create_graph=True)
        """

        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.cfg.gradient_clip,
            gradient_clip_algorithm="value",
        )

        optimizer.step()

        # memory leak issue with create_graph=True in backward https://github.com/pytorch/pytorch/issues/7343
        optimizer.zero_grad()

        # If the network is discontinuous, add smoothing.
        smooth_discontinuous_network(self, factor=self.cfg.factor)

    def on_train_epoch_end(self, outputs=None):
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["train_loss"])
        torch.cuda.empty_cache()

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
        elif self.cfg.optimizer.name == "adamw":
            optimizer = torch.optim.AdamW(
                params=self.parameters(), lr=self.cfg.optimizer.lr
            )
        elif self.cfg.optimizer.name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                max_iter=self.cfg.optimizer.max_iter,
            )
        elif self.cfg.optimizer.name == "lion":
            optimizer = Lion(
                params=self.parameters(),
                lr=self.cfg.optimizer.lr,
            )
        elif self.cfg.optimizer.name == "sophia":
            optimizer = SophiaG(
                params=self.parameters(),
                lr=self.cfg.optimizer.lr,
                rho=self.cfg.optimizer.rho
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
