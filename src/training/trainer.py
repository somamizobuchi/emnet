"""Simple iteration-based trainer for EyeMovementNet with GradNorm."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from pathlib import Path
from tqdm import tqdm

from ..models.model import EyeMovementNet
from ..data.recon_dataset import ReconDataset
from ..utils.reconstruction import (
    stitch_frames_by_position,
    stitch_frames_by_position_bilinear,
)
from ..utils import visualization as viz


# Auxiliary loss component names balanced by GradNorm (order matches log_weights indices)
AUX_TASK_NAMES = ["kernel_var", "regularization", "temporal_smoothness"]


class Trainer:
    """
    Simple iteration-based trainer for EyeMovementNet.

    Uses GradNorm (Chen et al., 2018) to dynamically balance loss weights
    based on gradient magnitudes. ALM (firing rate constraint) is kept
    separate with fixed weight since it has its own Lagrange multiplier dynamics.
    """

    def __init__(
        self,
        model: EyeMovementNet,
        dataset: ReconDataset,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        grad_norm_alpha: float = 1.5,
        device: str = "cpu",
        log_dir: Optional[str] = None,
        save_every: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.grad_norm_alpha = grad_norm_alpha

        # Data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.data_iter = iter(self.dataloader)

        # GradNorm: learnable log-weights for 4 loss components
        self.log_weights = nn.Parameter(torch.zeros(len(AUX_TASK_NAMES), device=device))

        # Optimizer with separate learning rates for gain/bias (more sensitive)
        param_groups = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "log_gain" not in n and "log_bias" not in n
                ],
                "lr": learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "log_gain" in n or "log_bias" in n
                ],
                "lr": learning_rate * 0.1,
            },
        ]
        self.optimizer = torch.optim.Adam(param_groups)

        # Separate optimizer for GradNorm weights
        self.weight_optimizer = torch.optim.Adam([self.log_weights], lr=0.025)

        # Initial loss values for relative training rate (set after first iteration)
        self.initial_losses: Optional[torch.Tensor] = None

        # Shared layer for GradNorm gradient computation
        self.shared_layer = self.model.frame_decoder.decoder.weight

        # Tensorboard logging
        self.writer = SummaryWriter(log_dir) if log_dir else None

        # Checkpoint saving
        self.save_every = save_every
        if save_every is not None:
            self.checkpoint_dir = Path(checkpoint_dir or log_dir or "checkpoints")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.iteration = 0

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model and optimizer state to a checkpoint file."""
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_{self.iteration:08d}.pt"
        torch.save(
            {
                "iteration": self.iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "log_weights": self.log_weights.data,
                "weight_optimizer_state_dict": self.weight_optimizer.state_dict(),
                "initial_losses": self.initial_losses,
            },
            path,
        )

    def _get_batch(self):
        """Get next batch, restarting iterator if needed."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        frames, target_img, eye_trace, mask, _ = batch
        frames = frames.to(self.device)
        target_img = target_img.to(self.device)
        eye_trace = eye_trace.to(self.device)
        mask = mask.to(self.device)

        return frames, target_img, eye_trace, mask

    def _compute_grad_norms(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute gradient norms of each weighted loss w.r.t. all model parameters."""
        weights = torch.exp(self.log_weights)
        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        norms = []
        for i, loss in enumerate(losses):
            weighted = weights[i] * loss
            grads = torch.autograd.grad(
                weighted, shared_params,
                retain_graph=True, create_graph=True, allow_unused=True,
            )
            total_norm = sum(
                (g.norm() ** 2 for g in grads if g is not None),
                torch.zeros(1, device=self.device),
            )
            norms.append(total_norm.sqrt())
        return torch.stack(norms)

    def train_step(self) -> dict:
        """Execute one training iteration."""
        self.model.train()
        self.optimizer.zero_grad()
        self.weight_optimizer.zero_grad()

        # Get batch
        frames, target_img, eye_trace, mask = self._get_batch()

        # Forward pass
        reconstructed, rgc_output, _ = self.model(frames, return_intermediates=True)

        # Align eye trace with temporal reduction
        batch_size = reconstructed.shape[0]
        pad_start = self.model.get_temporal_reduction()
        eye_trace_aligned = eye_trace[:, :, pad_start:]

        # Compute reconstruction loss
        batch_sse = []
        last_stitched = None
        for i in range(batch_size):
            trace = eye_trace_aligned[i].float()
            trace += torch.randn_like(trace) * 0.5
            eye_trace_cpu = trace.cpu()
            reconstructed_cpu = reconstructed[i].cpu()

            stitched_cpu = stitch_frames_by_position_bilinear(
                eye_trace_cpu, reconstructed_cpu, self.model.img_size
            )

            stitched = stitched_cpu.to(self.device)
            error = (target_img[i] - stitched) ** 2
            sse = (error * mask[i]).sum()
            num_pixels = mask[i].sum() + 1e-8
            mse = sse / num_pixels

            batch_sse.append(mse)
            last_stitched = stitched

        recon_loss = torch.stack(batch_sse).mean()

        # Compute other loss components
        alm_loss, constraint_violation = self.model.compute_firing_rate_loss(rgc_output)
        kernel_var_loss = self.model.compute_spatial_variance_loss()
        reg_loss = self.model.compute_regularization_loss()
        temporal_smoothness_loss = self.model.compute_temporal_smoothness_loss()

        # Auxiliary losses balanced by GradNorm (order matches AUX_TASK_NAMES)
        aux_losses = [kernel_var_loss, reg_loss, temporal_smoothness_loss]

        # Store initial losses for relative training rate
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(
                [l.detach().item() for l in aux_losses], device=self.device
            )
            self.initial_losses = torch.clamp(self.initial_losses, min=1e-8)

        # GradNorm: compute gradient norms and update weights for aux tasks
        grad_norms = self._compute_grad_norms(aux_losses)

        # Relative inverse training rates
        with torch.no_grad():
            current_losses = torch.tensor(
                [l.item() for l in aux_losses], device=self.device
            )
            relative_rates = current_losses / self.initial_losses
            relative_rates = relative_rates / relative_rates.mean()

        # GradNorm targets
        target_norms = grad_norms.detach().mean() * (
            relative_rates ** self.grad_norm_alpha
        )

        # GradNorm loss
        grad_norm_loss = (grad_norms - target_norms).abs().sum()

        # Update weight optimizer
        grad_norm_loss.backward(retain_graph=True)
        self.weight_optimizer.step()

        # Renormalize log_weights to keep product = 1
        with torch.no_grad():
            self.log_weights.data -= self.log_weights.data.mean()

        # Total loss: fixed recon + GradNorm-weighted aux + ALM
        aux_weights = torch.exp(self.log_weights.detach())
        total_loss = recon_loss + sum(w * l for w, l in zip(aux_weights, aux_losses)) + alm_loss

        # Backward and step main optimizer
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Apply constraints
        self.model.normalize_kernels()
        self.model.update_lagrange_multiplier(constraint_violation)

        self.iteration += 1

        # Tensorboard logging
        if self.writer:
            if self.iteration % 100 == 0:
                self.writer.add_scalar(
                    "loss/total", float(total_loss.item()), self.iteration
                )
                self.writer.add_scalar(
                    "loss/reconstruction", float(recon_loss.item()), self.iteration
                )
                self.writer.add_scalar(
                    "loss/alm", float(alm_loss.item()), self.iteration
                )
                self.writer.add_scalar(
                    "loss/kernel_variance",
                    float(kernel_var_loss.item()),
                    self.iteration,
                )
                self.writer.add_scalar(
                    "loss/regularization", float(reg_loss.item()), self.iteration
                )
                self.writer.add_scalar(
                    "loss/temporal_smoothness",
                    float(temporal_smoothness_loss.item()),
                    self.iteration,
                )
                self.writer.add_scalar(
                    "loss/grad_norm", float(grad_norm_loss.item()), self.iteration
                )
                self.writer.add_scalar(
                    "constraint/violation_mean",
                    float(constraint_violation.mean().item()),
                    self.iteration,
                )
                self.writer.add_scalar(
                    "constraint/violation_max",
                    float(constraint_violation.abs().max().item()),
                    self.iteration,
                )

                # Log GradNorm weights
                effective_weights = torch.exp(self.log_weights.detach())
                for j, name in enumerate(AUX_TASK_NAMES):
                    self.writer.add_scalar(
                        f"gradnorm/weight_{name}",
                        float(effective_weights[j].item()),
                        self.iteration,
                    )

            # Log kernels and parameters periodically
            if self.iteration % 1000 == 0:
                spatial, temporal = self.model.get_rgc_weights()

                spatial_grid = viz.plot_spatial_kernels(spatial, self.model.roi_size)
                self.writer.add_image(
                    "rgc/spatial_kernels", spatial_grid, self.iteration
                )

                temporal_plot = viz.plot_temporal_kernels(
                    temporal,
                    title="RGC Temporal Kernels",
                    delay=self.model.rgc_encoder.D,
                )
                self.writer.add_image(
                    "rgc/temporal_kernels", temporal_plot, self.iteration
                )

                v1_temporal = self.model.v1_decoder.temporal_weights
                v1_temporal_plot = viz.plot_temporal_kernels(
                    v1_temporal,
                    title="V1 Temporal Kernels",
                    delay=self.model.v1_decoder.D,
                )
                self.writer.add_image(
                    "v1/temporal_kernels", v1_temporal_plot, self.iteration
                )

                log_gain = self.model.rgc_encoder.log_gain
                if torch.isfinite(log_gain).all():
                    self.writer.add_histogram(
                        "rgc/log_gain", log_gain, self.iteration
                    )

                log_bias = self.model.rgc_encoder.log_bias
                if torch.isfinite(log_bias).all():
                    self.writer.add_histogram(
                        "rgc/log_bias", log_bias, self.iteration
                    )

                lagrange = self.model.rgc_encoder.lagrange_multiplier
                if torch.isfinite(lagrange).all():
                    self.writer.add_histogram(
                        "rgc/lambda", lagrange, self.iteration
                    )

                if last_stitched is not None:

                    def norm_img(img):
                        return (img - img.min()) / (img.max() - img.min() + 1e-8)

                    self.writer.add_image(
                        "recon/target",
                        norm_img(target_img[-1]),
                        self.iteration,
                        dataformats="HW",
                    )
                    self.writer.add_image(
                        "recon/reconstructed",
                        norm_img(last_stitched),
                        self.iteration,
                        dataformats="HW",
                    )

        # Return metrics
        effective_weights = torch.exp(self.log_weights.detach())
        return {
            "iteration": self.iteration,
            "total_loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "alm_loss": float(alm_loss.item()),
            "kernel_var_loss": float(kernel_var_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "temporal_smoothness_loss": float(temporal_smoothness_loss.item()),
            "grad_norm_loss": float(grad_norm_loss.item()),
            "constraint_violation_mean": float(constraint_violation.mean().item()),
            "constraint_violation_max": float(constraint_violation.abs().max().item()),
            "weights": {
                name: float(effective_weights[j].item())
                for j, name in enumerate(AUX_TASK_NAMES)
            },
        }

    def train(self, max_iterations: int):
        """Train for a fixed number of iterations."""
        print(f"Starting training for {max_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"GradNorm alpha: {self.grad_norm_alpha}")
        print()

        pbar = tqdm(range(max_iterations), desc="Training")
        for _ in pbar:
            metrics = self.train_step()

            # Save checkpoint
            if self.save_every and self.iteration % self.save_every == 0:
                self.save_checkpoint()

            # Update progress bar
            pbar.set_postfix(
                {
                    "total": f"{metrics['total_loss']:.2f}",
                    "recon": f"{metrics['recon_loss']:.2f}",
                    "alm": f"{metrics['alm_loss']:.2f}",
                    "gnorm": f"{metrics['grad_norm_loss']:.2f}",
                    "constraint": f"{metrics['constraint_violation_mean']:+.4f}",
                }
            )

        if self.writer:
            self.writer.flush()
            self.writer.close()
            print(f"\nTensorboard logs saved to: {self.writer.log_dir}")

        print(f"\nTraining complete! Final iteration: {self.iteration}")
