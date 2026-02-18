"""Simple iteration-based trainer for EyeMovementNet."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from tqdm import tqdm

from ..models.model import EyeMovementNet
from ..data.recon_dataset import ReconDataset
from ..utils.reconstruction import stitch_frames_by_position, stitch_frames_by_position_bilinear
from ..utils import visualization as viz


class Trainer:
    """
    Simple iteration-based trainer for EyeMovementNet.

    No epoch management - just iterates for a fixed number of steps.
    """

    def __init__(
        self,
        model: EyeMovementNet,
        dataset: ReconDataset,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        centering_weight: float = 0.0,
        l2_weight: float = 0.0,
        device: str = "cpu",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: EyeMovementNet model to train
            dataset: ReconDataset for training
            batch_size: Batch size for data loading
            learning_rate: Learning rate for optimizer
            centering_weight: Weight for kernel spatial variance loss (default: 0.0)
            l2_weight: Weight for L2 regularization on V1/Frame weights (default: 0.0)
            device: Device to train on ("cpu" or "cuda")
            log_dir: Directory for tensorboard logs (default: None, no logging)
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.centering_weight = centering_weight
        self.l2_weight = l2_weight
        self.device = device

        # Data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.data_iter = iter(self.dataloader)

        # Optimizer with separate learning rates for gain/bias (more sensitive)
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters()
                          if "log_gain" not in n and "log_bias" not in n],
                "lr": learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters()
                          if "log_gain" in n or "log_bias" in n],
                "lr": learning_rate * 0.1,  # 10x lower learning rate for gain/bias
            },
        ]
        self.optimizer = torch.optim.Adam(param_groups)

        # Tensorboard logging
        self.writer = SummaryWriter(log_dir) if log_dir else None

        # Training state
        self.iteration = 0

    def _get_batch(self):
        """Get next batch, restarting iterator if needed."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        # Move to device
        frames, target_img, eye_trace, mask, _ = batch
        frames = frames.to(self.device)
        target_img = target_img.to(self.device)
        eye_trace = eye_trace.to(self.device)
        mask = mask.to(self.device)

        return frames, target_img, eye_trace, mask

    def train_step(self) -> dict:
        """
        Execute one training iteration.

        Returns:
            dict: Loss values and metrics for this step
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Get batch
        frames, target_img, eye_trace, mask = self._get_batch()

        # Check for NaN in input data
        if torch.isnan(frames).any() or torch.isnan(target_img).any() or torch.isnan(eye_trace).any():
            print(f"\n⚠️  NaN detected in input data at iteration {self.iteration}")
            print(f"   - frames has NaN: {torch.isnan(frames).any().item()}")
            print(f"   - target_img has NaN: {torch.isnan(target_img).any().item()}")
            print(f"   - eye_trace has NaN: {torch.isnan(eye_trace).any().item()}")
            raise ValueError("NaN in input data from dataset")

        # Forward pass
        reconstructed, rgc_output, _ = self.model(frames, return_intermediates=True)

        # Check for NaNs after forward pass
        if torch.isnan(reconstructed).any() or torch.isnan(rgc_output).any():
            print(f"\n⚠️  NaN detected after forward pass at iteration {self.iteration}")
            print(f"   - reconstructed has NaN: {torch.isnan(reconstructed).any().item()}")
            print(f"   - rgc_output has NaN: {torch.isnan(rgc_output).any().item()}")
            print(f"   - Stopping training to prevent corruption")
            raise ValueError("NaN detected in forward pass")

        # Reshape reconstructed frames: (batch, time, roi_size, roi_size)
        batch_size = reconstructed.shape[0]

        # Align eye trace with temporal reduction
        pad_start = self.model.get_temporal_reduction()
        eye_trace_aligned = eye_trace[:, :, pad_start:]

        # Compute reconstruction loss (SSE over batch and pixels)
        # Note: stitching is done on CPU as it uses integer indexing which is slow on MPS/CUDA
        batch_sse = []
        last_stitched = None
        for i in range(batch_size):
            # Move to CPU for stitching
            trace = eye_trace_aligned[i].float()
            trace += torch.randn_like(trace) * 0.5  # Add noise (reduced std to 0.5 pixels)
            eye_trace_cpu = trace.cpu()
            reconstructed_cpu = reconstructed[i].cpu()

            # Stitch on CPU
            # stitched_cpu = stitch_frames_by_position(
            #     eye_trace_cpu, reconstructed_cpu, self.model.img_size
            # )
            stitched_cpu = stitch_frames_by_position_bilinear(
                eye_trace_cpu, reconstructed_cpu, self.model.img_size
            )

            # Check for NaN in stitched result
            if torch.isnan(stitched_cpu).any():
                print(f"\n⚠️  NaN detected in stitching at iteration {self.iteration}, batch {i}")
                print(f"   - eye_trace range: [{trace.min().item():.2f}, {trace.max().item():.2f}]")
                print(f"   - reconstructed range: [{reconstructed_cpu.min().item():.2f}, {reconstructed_cpu.max().item():.2f}]")
                raise ValueError("NaN in stitching operation")

            # Move back to device for loss computation
            stitched = stitched_cpu.to(self.device)

            # Compute sum of squared errors for this sample (only on masked region)
            # Compute error only where mask is positive (visited regions)
            error = (target_img[i] - stitched) ** 2
            sse = (error * mask[i]).sum()

            # Normalize by number of pixels to get mean squared error
            num_pixels = mask[i].sum() + 1e-8  # Add epsilon to prevent division by zero
            mse = sse / num_pixels

            batch_sse.append(mse)
            last_stitched = stitched

        recon_loss = torch.stack(batch_sse).mean()

        # Compute firing rate constraint loss
        alm_loss, constraint_violation = self.model.compute_firing_rate_loss(rgc_output)

        # Compute kernel spatial variance loss
        kernel_var_loss = self.model.compute_spatial_variance_loss()

        # Compute L2 regularization loss
        l2_loss = self.model.compute_l2_loss()

        # Total loss
        total_loss = (
            recon_loss
            + alm_loss
            + self.centering_weight * kernel_var_loss
            + self.l2_weight * l2_loss
        )

        # Check for NaN in losses
        if torch.isnan(total_loss):
            print(f"\n⚠️  NaN detected in loss computation at iteration {self.iteration}")
            print(f"   - recon_loss: {recon_loss.item()}")
            print(f"   - alm_loss: {alm_loss.item()}")
            print(f"   - kernel_var_loss: {kernel_var_loss.item()}")
            print(f"   - l2_loss: {l2_loss.item()}")
            print(f"   - constraint_violation: {constraint_violation}")
            raise ValueError("NaN detected in loss")

        # Backward pass
        total_loss.backward()

        # Gradient clipping to prevent explosion (BEFORE NaN check)
        # This clips gradients but doesn't fix NaN, so we still need to detect them
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        # Check for NaN in gradients after clipping
        has_nan_grad = False
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"\n⚠️  NaN detected in gradient at iteration {self.iteration}")
                print(f"   - Parameter: {name}")
                print(f"   - Total gradient norm before clip: {total_grad_norm.item()}")
                print(f"   - Parameter value range: [{param.min().item():.4f}, {param.max().item():.4f}]")
                has_nan_grad = True
                break

        if has_nan_grad:
            # Zero out NaN gradients and continue (emergency recovery)
            print(f"   - Zeroing NaN gradients and continuing...")
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

        self.optimizer.step()

        # Clamp log_gain and log_bias to prevent runaway (after optimizer step)
        with torch.no_grad():
            self.model.rgc_encoder.log_gain.clamp_(-10.0, 10.0)
            self.model.rgc_encoder.log_bias.clamp_(-10.0, 10.0)

        # Apply constraints
        self.model.normalize_kernels()
        self.model.update_lagrange_multiplier(constraint_violation)

        # Check for NaNs in model parameters (optional debugging)
        if self.iteration % 100 == 0:
            nan_status = self.model.rgc_encoder.check_for_nans()
            if any(nan_status.values()):
                print(f"\n⚠️  WARNING: NaN detected at iteration {self.iteration}:")
                for param_name, has_nan in nan_status.items():
                    if has_nan:
                        print(f"  - {param_name}")

        self.iteration += 1

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar(
                "loss/total", float(total_loss.item()), self.iteration
            )
            self.writer.add_scalar(
                "loss/reconstruction", float(recon_loss.item()), self.iteration
            )
            self.writer.add_scalar("loss/alm", float(alm_loss.item()), self.iteration)
            self.writer.add_scalar(
                "loss/kernel_variance", float(kernel_var_loss.item()), self.iteration
            )
            self.writer.add_scalar(
                "loss/l2_regularization", float(l2_loss.item()), self.iteration
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
            self.writer.add_scalar(
                "train/grad_norm", float(total_grad_norm.item()), self.iteration
            )

            # Log kernels and parameters periodically
            if self.iteration % 100 == 0:
                spatial, temporal = self.model.get_rgc_weights()

                # Spatial kernels grid
                spatial_grid = viz.plot_spatial_kernels(spatial, self.model.roi_size)
                self.writer.add_image(
                    "rgc/spatial_kernels", spatial_grid, self.iteration
                )

                # Temporal kernels plot
                temporal_plot = viz.plot_temporal_kernels(temporal)
                self.writer.add_image(
                    "rgc/temporal_kernels", temporal_plot, self.iteration
                )

                # Histograms of parameters (only if valid)
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

                # Log images (normalized to [0, 1] for visualization)
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
        return {
            "iteration": self.iteration,
            "total_loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "alm_loss": float(alm_loss.item()),
            "kernel_var_loss": float(kernel_var_loss.item()),
            "l2_loss": float(l2_loss.item()),
            "constraint_violation_mean": float(constraint_violation.mean().item()),
            "constraint_violation_max": float(constraint_violation.abs().max().item()),
        }

    def train(self, max_iterations: int):
        """
        Train for a fixed number of iterations.

        Args:
            max_iterations: Maximum number of training iterations
        """
        print(f"Starting training for {max_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print()

        pbar = tqdm(range(max_iterations), desc="Training")
        for _ in pbar:
            metrics = self.train_step()

            # Update progress bar with metrics
            pbar.set_postfix(
                {
                    "total": f"{metrics['total_loss']:.2f}",
                    "recon": f"{metrics['recon_loss']:.2f}",
                    "alm": f"{metrics['alm_loss']:.2f}",
                    "kvar": f"{metrics['kernel_var_loss']:.2f}",
                    "l2": f"{metrics['l2_loss']:.2f}",
                    "constraint": f"{metrics['constraint_violation_mean']:+.4f}",
                }
            )

        if self.writer:
            self.writer.flush()
            self.writer.close()
            print(f"\nTensorboard logs saved to: {self.writer.log_dir}")

        print(f"\nTraining complete! Final iteration: {self.iteration}")
