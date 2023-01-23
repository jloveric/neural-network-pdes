from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from neural_network_pdes.euler_networks import ImageSampler, generate_images, Net
from pytorch_lightning.callbacks import LearningRateMonitor


@hydra.main(config_path="../config", config_name="euler")
def run_implicit_images(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch:03d}", monitor="train_loss"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            callbacks=[checkpoint_callback, ImageSampler(cfg=cfg), lr_monitor],
        )
        model = Net(cfg)
        trainer.fit(model)
        print("testing")
        # trainer.test(model)

        print("finished testing")
        print("best check_point", trainer.checkpoint_callback.best_model_path)
    else:
        # plot some data
        print("evaluating result")
        print("cfg.checkpoint", cfg.checkpoint)

        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        print("checkpoint_path", checkpoint_path)

        model = Net.load_from_checkpoint(checkpoint_path)

        generate_images(
            model=model,
            save_to="file" if cfg.save_images else None,
            layer_type=cfg.mlp.layer_type,
        )


if __name__ == "__main__":
    run_implicit_images()
