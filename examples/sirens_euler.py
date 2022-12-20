from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from high_order_layers_torch.positional_embeddings import ClassicSinusoidalEmbedding
from neural_network_pdes.euler_networks import (
    ImageSampler,
    generate_images,
    nonlinearity_options,
    PDEDataset,
    euler_loss,
)
import torch
from torch.nn import LazyBatchNorm1d
from functorch import vmap, jacrev
import math


class ClassicSinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        """
        Traditional positional embedding
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # print('embedding.x.shape', x.shape)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # print('emb.shape', emb.shape)
        return emb


class SirensNet(LightningModule):
    def __init__(self, cfg: DictConfig):
        # TODO : didn't want to copy this, only slightly different from NET
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

        self._embedding_layer = ClassicSinusoidalEmbedding(cfg.mlp.embedding_size).to(
            self.device
        )

        self._mlp = LowOrderMLP(
            in_width=cfg.mlp.embedding_size * cfg.mlp.input.width,
            out_width=cfg.mlp.output.width,
            hidden_width=cfg.mlp.hidden.width,
            hidden_layers=cfg.mlp.hidden.layers,
            normalization=None if cfg.mlp.normalize is False else LazyBatchNorm1d,
            non_linearity=nl,
        )

        # self.model = nn.Sequential(embedding_layer, mlp)

        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self._embedding_layer(x)
        xm = x.reshape(x.shape[0], -1)
        y = self._mlp(xm)
        return y

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

        """
        for inp in x:
            inp = inp.unsqueeze(dim=0)
            jacobian = torch.autograd.functional.jacobian(
                self, inp, create_graph=True, strict=False, vectorize=True
            )
            jacobian_list.append(jacobian)
        """

        # gradients = torch.reshape(torch.stack(jacobian_list), (-1, 3, 2))
        # print('gradients.shape', gradients.shape)
        # This currently fails, due to a bug in functorch
        # TODO: re-enable this in the future.
        # print('x pre.shape', x.shape)
        jacobian = vmap(jacrev(self.forward))(x.reshape(x.shape[0], 1, x.shape[1]))
        nj = jacobian.reshape(-1, 3, 2)
        # print('jacobian', jacobian.shape)
        # exit(0)
        in_loss, ic_loss, left_bc_loss, right_bc_loss = euler_loss(
            x=x, q=y_hat, grad_q=nj, targets=y
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


@hydra.main(config_path="../config", config_name="euler-sirens")
def run_implicit_images(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch:03d}", monitor="train_loss"
        )
        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            callbacks=[checkpoint_callback, ImageSampler(cfg=cfg)],
        )
        model = SirensNet(cfg)
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
        print("checkpoint_path", checkpoint_path)

        model = SirensNet.load_from_checkpoint(checkpoint_path)

        generate_images(
            model=model,
            save_to="file" if cfg.save_images else None,
            layer_type=cfg.mlp.layer_type,
        )


if __name__ == "__main__":
    run_implicit_images()
