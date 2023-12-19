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


@hydra.main(config_path="../config", config_name="euler", version_base="1.3")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch:03d}", monitor="train_loss"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        if cfg.refinement.type == None:
            trainer = Trainer(
                max_epochs=cfg.max_epochs,
                accelerator=cfg.accelerator,
                callbacks=[checkpoint_callback, ImageSampler(cfg=cfg), lr_monitor],
            )
            model = Net(cfg)
            trainer.fit(model)
            print("testing")
            # trainer.test(model)

            print("finished testing")
            print("best check_point", trainer.checkpoint_callback.best_model_path)

        elif cfg.refinement.type == "p_refine":
            # diff = cfg.mlp.target_n - cfg.mlp.n
            model = Net(cfg)
            cfg.mlp.n = cfg.refinement.start_n
            n = cfg.mlp.n
            cfg.mlp.n_in = n
            cfg.mlp.n_out = n
            cfg.mlp.n_hidden = n
            for order in range(
                cfg.refinement.start_n,
                cfg.refinement.target_n + 1,
                cfg.refinement.step,
            ):
                trainer = Trainer(
                    max_epochs=cfg.refinement.epochs
                    if order < cfg.refinement.target_n
                    else cfg.max_epochs,
                    accelerator=cfg.accelerator,
                    callbacks=[checkpoint_callback, ImageSampler(cfg=cfg), lr_monitor],
                )
                # trainer = Trainer(max_epochs=cfg.max_epochs // diff, gpus=cfg.gpus)
                print(f"Training order {order}")
                trainer.fit(model)
                # trainer.test(model)
                n = order + cfg.refinement.step
                cfg.mlp.n = n
                cfg.mlp.n_in = n
                cfg.mlp.n_out = n
                cfg.mlp.n_hidden = n

                next_model = Net(cfg)

                interpolate_high_order_mlp(network_in=model, network_out=next_model)
                model = next_model

        else:
            print(f"Refinement type not recognized {cfg.refinement.type}")
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
    run()
