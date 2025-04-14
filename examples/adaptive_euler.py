from typing import List
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from non_uniform_piecewise_layers import AdaptivePiecewiseMLP
from neural_network_pdes.euler_networks import ImageSampler, generate_images
from neural_network_pdes.common import pde_grid, solution_points
from neural_network_pdes import euler as pform
from neural_network_pdes import euler_conservative as cform
from lion_pytorch import Lion

logger = logging.getLogger(__name__)


class AdaptiveNet(nn.Module):
    """Neural network for solving Euler equations using AdaptivePiecewiseMLP."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        self._gamma = cfg.physics.gamma
        
        # Create the AdaptivePiecewiseMLP
        # Input dimension is 2 (x, t), output dimension is 3 (density, velocity, pressure)
        widths = [2] + cfg.network.hidden_layers + [3]
        
        self.model = AdaptivePiecewiseMLP(
            width=widths,
            num_points=cfg.network.num_points,
            position_range=cfg.network.position_range,
            anti_periodic=cfg.network.anti_periodic,
            position_init=cfg.network.position_init,
            normalization=cfg.network.normalization
        )
        
        # Track the number of move_smoothest operations performed
        self.move_smoothest_count = 0
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def flux(self, x):
        """Compute the flux for the Euler equations."""
        # This is a placeholder for compatibility with the original code
        # Not used in this implementation
        return None
    
    def move_smoothest(self):
        """Apply the move_smoothest operation to the network."""
        if self.cfg.adaptive.move_smoothest:
            moved_pairs, total_pairs = self.model.move_smoothest(weighted=True, threshold=self.cfg.adaptive.threshold)
            self.move_smoothest_count += 1
            logger.info(f"Move smoothest: {moved_pairs}/{total_pairs} pairs moved")
            return moved_pairs, total_pairs
        return 0, 0


class AdaptiveEulerTrainer:
    """Trainer for the Euler equations using AdaptivePiecewiseMLP."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.accelerator if torch.cuda.is_available() else "cpu")
        
        # Create the model
        self.model = AdaptiveNet(cfg).to(self.device)
        
        # Create the optimizer
        if cfg.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        elif cfg.optimizer == "lion":
            self.optimizer = Lion(self.model.parameters(), lr=cfg.learning_rate)
        else:
            raise ValueError(f"Optimizer {cfg.optimizer} not supported")
        
        # Create the dataset
        x, t = solution_points(cfg.data_size)
        
        # Initial condition
        rho = torch.where(x < 0, 1.0, 0.125)
        p = torch.where(x < 0, 1.0, 0.1)
        u = torch.where(x < 0, 0.0, 0.0)
        
        self.inputs = torch.cat([x, t], dim=1).to(self.device)
        self.targets = torch.cat([rho, u, p], dim=1).to(self.device)
        
        # Create the dataloader
        indices = torch.randperm(len(self.inputs))
        self.inputs = self.inputs[indices]
        self.targets = self.targets[indices]
        
        # Track the best loss
        self.best_loss = float('inf')
        self.best_model_path = None
        
    def compute_gradients(self, x, q):
        """Compute the gradients of the network output with respect to the input."""
        x.requires_grad_(True)
        q_pred = self.model(x)
        
        # Compute gradients
        grad_outputs = torch.ones_like(q_pred)
        grad_q = torch.autograd.grad(
            outputs=q_pred,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Reshape gradients
        grad_q = grad_q.reshape(q_pred.shape[0], q_pred.shape[1], x.shape[1])
        
        x.requires_grad_(False)
        return q_pred, grad_q
    
    def compute_loss(self, x, q_pred, grad_q):
        """Compute the loss for the Euler equations."""
        if self.cfg.form == "conservative":
            in_loss, ic_loss, left_bc_loss, right_bc_loss = cform.euler_loss(
                x=x,
                q=q_pred,
                grad_q=grad_q,
                targets=self.targets,
                eps=self.cfg.physics.artificial_viscosity,
                time_decay=self.cfg.time_decay,
                scale_x=self.cfg.scale_x,
                scale_t=self.cfg.scale_t,
                solve_waves=self.cfg.solve_waves,
            )
        else:
            in_loss, ic_loss, left_bc_loss, right_bc_loss = pform.euler_loss(
                x=x,
                q=q_pred,
                grad_q=grad_q,
                targets=self.targets,
                eps=self.cfg.physics.artificial_viscosity,
                time_decay=self.cfg.time_decay,
                scale_x=self.cfg.scale_x,
                scale_t=self.cfg.scale_t,
                solve_waves=self.cfg.solve_waves,
            )
        
        # Compute the total loss
        total_loss = (
            self.cfg.loss_weight.interior * in_loss
            + self.cfg.loss_weight.initial * ic_loss
            + self.cfg.loss_weight.boundary * left_bc_loss
            + self.cfg.loss_weight.boundary * right_bc_loss
        )
        
        return total_loss, in_loss, ic_loss, left_bc_loss, right_bc_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        # Process in batches
        batch_size = self.cfg.batch_size
        num_batches = (len(self.inputs) + batch_size - 1) // batch_size
        
        total_loss = 0.0
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.inputs))
            x_batch = self.inputs[start_idx:end_idx]
            
            # Forward pass and compute gradients
            self.optimizer.zero_grad()
            q_pred, grad_q = self.compute_gradients(x_batch, None)
            
            # Compute loss
            loss, in_loss, ic_loss, left_bc_loss, right_bc_loss = self.compute_loss(
                x_batch, q_pred, grad_q
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.gradient_clip
                )
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Apply move_smoothest operation
            if (
                self.cfg.adaptive.enabled
                and self.cfg.adaptive.move_smoothest
                and epoch >= self.cfg.adaptive.move_smoothest_after
                and batch_idx % self.cfg.adaptive.frequency == 0
            ):
                self.model.move_smoothest()
        
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save a checkpoint of the model."""
        checkpoint_dir = os.getcwd()
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'move_smoothest_count': self.model.move_smoothest_count,
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            self.best_model_path = best_model_path
    
    def train(self):
        """Train the model for the specified number of epochs."""
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model: {self.model}")
        
        for epoch in tqdm(range(self.cfg.max_epochs)):
            # Train for one epoch
            avg_loss = self.train_epoch(epoch)
            
            # Log progress
            logger.info(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
            
            # Save checkpoint
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            if epoch % self.cfg.save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, avg_loss, is_best)
                
                # Generate images
                if self.cfg.save_images:
                    self.generate_images(epoch)
        
        logger.info("Training completed!")
        logger.info(f"Best loss: {self.best_loss:.6f}")
        logger.info(f"Best model saved at: {self.best_model_path}")
        
        # Generate final images
        if self.cfg.save_images:
            self.generate_images(self.cfg.max_epochs)
    
    def generate_images(self, epoch):
        """Generate and save images of the current solution."""
        self.model.eval()
        
        # Create directory for images
        images_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Generate grid for visualization
        grid = pde_grid().to(self.device)
        
        with torch.no_grad():
            y_hat = self.model(grid).cpu().numpy()
        
        outputs = y_hat.reshape(100, 100, 3)
        names = ["Density", "Velocity", "Pressure"]
        
        for j, name in enumerate(names):
            plt.figure(j + 1)
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot colormap
            c = ax0.pcolor(outputs[:, :, j])
            fig.colorbar(c, ax=ax0)
            ax0.set_xlabel("x")
            ax0.set_ylabel("time")
            
            # Plot line slices at different times
            for i in range(0, 100, 20):
                ax1.plot(outputs[:, i, j], label=f"t={i/100:.2f}")
            
            ax1.set_xlabel("x")
            ax1.set_ylabel(f"{name}")
            ax1.legend()
            
            ax0.set_title(f"{name} at Epoch {epoch}")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(images_dir, f"{name}_epoch_{epoch:04d}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


@hydra.main(config_path="../config", config_name="adaptive_euler", version_base="1.3")
def run(cfg: DictConfig):
    """Main function to run the training or evaluation."""
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Original working directory : {hydra.utils.get_original_cwd()}")
    
    if cfg.train:
        # Create and train the model
        trainer = AdaptiveEulerTrainer(cfg)
        trainer.train()
    else:
        # Load and evaluate a trained model
        print("Evaluating result")
        print("cfg.checkpoint", cfg.checkpoint)
        
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        print("checkpoint_path", checkpoint_path)
        
        # Create the model
        model = AdaptiveNet(cfg)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate images
        device = torch.device(cfg.accelerator if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create directory for images
        images_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Generate grid for visualization
        grid = pde_grid().to(device)
        
        with torch.no_grad():
            y_hat = model(grid).cpu().numpy()
        
        outputs = y_hat.reshape(100, 100, 3)
        names = ["Density", "Velocity", "Pressure"]
        
        for j, name in enumerate(names):
            plt.figure(j + 1)
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot colormap
            c = ax0.pcolor(outputs[:, :, j])
            fig.colorbar(c, ax=ax0)
            ax0.set_xlabel("x")
            ax0.set_ylabel("time")
            
            # Plot line slices at different times
            for i in range(0, 100, 20):
                ax1.plot(outputs[:, i, j], label=f"t={i/100:.2f}")
            
            ax1.set_xlabel("x")
            ax1.set_ylabel(f"{name}")
            ax1.legend()
            
            ax0.set_title(f"{name} (Evaluation)")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(images_dir, f"{name}_evaluation.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    run()
