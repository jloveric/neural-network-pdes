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
from torch.nn import LazyBatchNorm1d
import io
import PIL.Image
from torchvision import transforms
import functorch

from neural_network_pdes.euler import pde_grid, PDEDataset, euler_loss


class SinLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


nonlinearity_options = {"relu": torch.nn.ReLU(), "sin": SinLayer()}


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # TODO: Add a fixed layer after the input layer that
        # produces [x, t, (x+t), (x-t)] with no weight adjustment
        # this would be a fixed linear transform layer.

        nl = (
            None
            if cfg.mlp.nonlinearity is None
            else nonlinearity_options[cfg.mlp.nonlinearity]
        )

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
        )
        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def setup(self, stage: int):
        self.train_dataset = PDEDataset(
            size=self.cfg.data_size, rotations=self.cfg.rotations
        )
        self.test_dataset = PDEDataset(
            size=self.cfg.data_size, rotations=self.cfg.rotations
        )

    def training_step(self, batch: Tensor, batch_idx: int):
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

        # This currently fails, due to a bug in functorch
        # TODO: re-enable this in the future.
        # jacobian = functorch.vmap(functorch.jacrev(self))(x)

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
            drop_last=True
        )
        return trainloader

    def test_dataloader(self):

        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=True
        )
        return testloader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


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

    if save_to != "memory":
        plt.show()
    else:
        return image_list

    return None
