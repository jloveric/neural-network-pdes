"""
Adaptive Euler equations solver using non-uniform piecewise layers.
This version uses AdaptivePiecewiseMLP from non_uniform_piecewise_layers
without PyTorch Lightning.
"""
import os
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.func import jacrev
from torch import vmap
from lion_pytorch import Lion
import imageio

from non_uniform_piecewise_layers import AdaptivePiecewiseMLP
from non_uniform_piecewise_layers.utils import largest_error

import neural_network_pdes.euler as pform
import neural_network_pdes.euler_conservative as cform
from neural_network_pdes.common import pde_grid, solution_points

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def get_device(accelerator: str) -> torch.device:
    """Get the appropriate device based on config."""
    if accelerator == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataset(cfg: DictConfig, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Create the PDE dataset for training."""
    if cfg.form == "conservative":
        dataset = cform.PDEDataset(size=cfg.data_size)
    else:
        dataset = pform.PDEDataset(size=cfg.data_size)
    
    inputs = dataset.input.to(device)
    targets = dataset.output.to(device)
    return inputs, targets


def compute_flux(model: nn.Module, x: Tensor, gamma: float) -> Tensor:
    """Compute flux for conservative form."""
    return cform.flux(model(x), gamma=gamma)


def compute_loss(
    model: nn.Module,
    x: Tensor,
    targets: Tensor,
    cfg: DictConfig,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the PDE loss including interior, initial condition, and boundary losses.
    Returns total loss and individual components.
    """
    x.requires_grad_(True)
    y_hat = model(x)
    
    # Clamp density and pressure to positive values
    y_hat = y_hat.clone()
    y_hat[:, 0] = torch.clamp(y_hat[:, 0], min=0.01)
    y_hat[:, 2] = torch.clamp(y_hat[:, 2], min=0.01)
    
    # Compute Jacobian
    xf = x.reshape(x.shape[0], 1, x.shape[1])
    jacobian = vmap(jacrev(model))(xf)
    nj = jacobian.reshape(-1, 3, 2)
    
    if cfg.form == "conservative":
        # Compute flux jacobian for conservative form
        def flux_fn(inp):
            return cform.flux(model(inp), gamma=cfg.physics.gamma)
        
        flux_jacobian = vmap(jacrev(flux_fn))(xf).reshape(-1, 3, 2)
        
        in_loss, ic_loss, left_bc_loss, right_bc_loss = cform.euler_loss(
            x=x,
            q=y_hat,
            grad_q=nj,
            grad_f=flux_jacobian,
            hessian=None,
            artificial_viscosity=cfg.physics.artificial_viscosity,
            targets=targets,
            eps=cfg.loss_weight.discontinuity,
            scale_x=cfg.scale_x,
            scale_t=cfg.scale_t,
        )
    else:  # primitive form
        in_loss, ic_loss, left_bc_loss, right_bc_loss = pform.euler_loss(
            x=x,
            q=y_hat,
            grad_q=nj,
            targets=targets,
            eps=cfg.loss_weight.discontinuity,
            time_decay=cfg.time_decay,
            scale_x=cfg.scale_x,
            scale_t=cfg.scale_t,
            solve_waves=cfg.solve_waves,
        )
    
    total_loss = (
        cfg.loss_weight.interior * in_loss
        + cfg.loss_weight.initial * ic_loss
        + cfg.loss_weight.boundary * (left_bc_loss + right_bc_loss)
    )
    
    return total_loss, in_loss, ic_loss, left_bc_loss, right_bc_loss, y_hat


def create_optimizer(parameters, cfg: DictConfig):
    """Create optimizer based on config."""
    if cfg.optimizer.name == "adam":
        return optim.Adam(parameters, lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == "adamw":
        return optim.AdamW(parameters, lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == "lion":
        return Lion(parameters, lr=cfg.optimizer.lr)
    else:
        return optim.Adam(parameters, lr=cfg.optimizer.lr)


def generate_images(
    model: nn.Module,
    device: torch.device,
    epoch: int,
    save_to: str = "file",
    layer_type: str = "adaptive",
) -> None:
    """Generate and save visualization images."""
    model.eval()
    inputs = pde_grid().detach().to(device)
    
    with torch.no_grad():
        y_hat = model(inputs).cpu().numpy()
    
    outputs = y_hat.reshape(100, 100, 3)
    names = ["Density", "Velocity", "Pressure"]
    
    for j, name in enumerate(names):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8))
        
        c = ax0.pcolor(outputs[:, :, j])
        ax0.set_xlabel("x")
        ax0.set_ylabel("time")
        plt.colorbar(c, ax=ax0)
        
        for i in range(0, 100, 20):
            ax1.plot(outputs[:, i, j], label=f"t={i}")
        
        ax1.set_xlabel("x")
        ax1.set_ylabel(f"{name}")
        ax1.legend()
        
        ax0.set_title(f"{name} with {layer_type} layers - Epoch {epoch}")
        
        if save_to == "file":
            plt.savefig(f"{name.lower()}_epoch_{epoch:04d}.png", dpi=150)
        
        plt.close()
    
    model.train()


def save_piecewise_plots(model: nn.Module, epoch: int) -> None:
    """Save plots showing the piecewise approximations for each layer."""
    num_layers = len(model.layers)
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 5 * num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx, (layer, ax) in enumerate(zip(model.layers, axes)):
        positions = layer.positions.data.cpu()
        values = layer.values.data.cpu()
        
        max_lines = 10
        for in_dim in range(min(positions.shape[0], max_lines)):
            for out_dim in range(min(positions.shape[1], max_lines)):
                pos = positions[in_dim, out_dim].numpy()
                val = values[in_dim, out_dim].numpy()
                ax.plot(pos, val, 'o-', alpha=0.5, markersize=2)
        
        ax.set_title(f'Layer {layer_idx + 1} Piecewise Approximations')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(f'weights_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_convergence_plot(losses: list, epochs: list) -> None:
    """Save a plot showing the convergence of loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(epochs, losses, 'b-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.grid(True)
    ax.set_title('Convergence Plot')
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    plt.close()


@hydra.main(config_path="../config", config_name="euler_adaptive", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Output directory: {HydraConfig.get().run.dir}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    device = get_device(cfg.accelerator)
    logger.info(f"Using device: {device}")
    
    # Create dataset
    inputs, targets = create_dataset(cfg, device)
    
    # Create model
    model = AdaptivePiecewiseMLP(
        width=cfg.model.width,
        num_points=cfg.model.num_points,
        position_range=tuple(cfg.model.position_range),
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model.parameters(), cfg)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=cfg.optimizer.patience,
        factor=cfg.optimizer.factor,
        verbose=True,
    )
    
    # Create GIF writers for visualization
    progress_writers = {
        name.lower(): imageio.get_writer(
            f'{name.lower()}_progress.gif', mode='I', duration=cfg.visualization.gif_duration
        )
        for name in ["Density", "Velocity", "Pressure"]
    }
    weights_writer = imageio.get_writer('weights.gif', mode='I', duration=cfg.visualization.gif_duration)
    
    # Training tracking
    losses = []
    epochs_list = []
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(cfg.max_epochs):
        model.train()
        
        # Sample a batch
        batch_indices = torch.randperm(len(inputs))[:cfg.batch_size]
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]
        
        # Compute loss
        total_loss, in_loss, ic_loss, left_bc_loss, right_bc_loss, predictions = compute_loss(
            model, batch_inputs, batch_targets, cfg
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_value_(model.parameters(), cfg.gradient_clip)
        
        optimizer.step()
        
        # Record loss
        loss_value = total_loss.item()
        losses.append(loss_value)
        epochs_list.append(epoch)
        
        # Update learning rate scheduler
        scheduler.step(loss_value)
        
        # Adaptive refinement
        if cfg.training.adapt == "global_error":
            with torch.no_grad():
                error = torch.abs(predictions - batch_targets)
                new_value = largest_error(error, batch_inputs)
                if new_value is not None:
                    success = model.remove_add(new_value)
                    if success:
                        optimizer = create_optimizer(model.parameters(), cfg)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            patience=cfg.optimizer.patience,
                            factor=cfg.optimizer.factor,
                            verbose=True,
                        )
        elif cfg.training.adapt == "move":
            threshold = cfg.training.move_threshold
            moved_pairs, total_pairs = model.move_smoothest(
                weighted=cfg.training.weighted, threshold=threshold
            )
            if moved_pairs > 0:
                logger.info(f'Moved {moved_pairs}/{total_pairs} pairs ({moved_pairs/total_pairs*100:.2f}%)')
                optimizer = create_optimizer(model.parameters(), cfg)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=cfg.optimizer.patience,
                    factor=cfg.optimizer.factor,
                    verbose=True,
                )
        
        # Save best model
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }, 'best_model.pt')
        
        # Logging and visualization
        if epoch % cfg.visualization.plot_interval == 0:
            logger.info(
                f'Epoch {epoch}/{cfg.max_epochs}, '
                f'Loss: {loss_value:.6f}, '
                f'Interior: {in_loss.item():.6f}, '
                f'IC: {ic_loss.item():.6f}, '
                f'BC: {(left_bc_loss + right_bc_loss).item():.6f}'
            )
            
            # Generate and save images
            generate_images(model, device, epoch, save_to="file", layer_type="adaptive")
            
            # Add frames to GIFs
            for name in ["density", "velocity", "pressure"]:
                img_path = f'{name}_epoch_{epoch:04d}.png'
                if os.path.exists(img_path):
                    progress_writers[name].append_data(imageio.imread(img_path))
            
            # Save piecewise plots
            save_piecewise_plots(model, epoch)
            weights_writer.append_data(imageio.imread(f'weights_epoch_{epoch:04d}.png'))
    
    # Save final convergence plot
    save_convergence_plot(losses, epochs_list)
    
    # Close GIF writers
    for writer in progress_writers.values():
        writer.close()
    weights_writer.close()
    
    logger.info(f"Training complete. Best loss: {best_loss:.6f}")
    logger.info("GIFs and plots saved successfully!")


if __name__ == "__main__":
    main()
