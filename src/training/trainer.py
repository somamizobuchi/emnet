"""Simple iteration-based trainer for EyeMovementNet."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from tqdm import tqdm

from ..models.model import EyeMovementNet
from ..data.recon_dataset import ReconDataset
from ..utils.reconstruction import stitch_frames_by_position


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
            device: Device to train on ("cpu" or "cuda")
            log_dir: Directory for tensorboard logs (default: None, no logging)
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        # Data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.data_iter = iter(self.dataloader)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        # Forward pass
        reconstructed, rgc_output, _ = self.model(frames, return_intermediates=True)

        # Reshape reconstructed frames: (batch, time, roi_size, roi_size)
        batch_size = reconstructed.shape[0]

        # Align eye trace with temporal reduction
        pad_start = self.model.get_temporal_reduction()
        eye_trace_aligned = eye_trace[:, :, pad_start:]

        # Compute reconstruction loss (sum over batch)
        # Note: stitching is done on CPU as it uses integer indexing which is slow on MPS/CUDA
        recon_loss = 0.0
        for i in range(batch_size):
            # Move to CPU for stitching
            eye_trace_cpu = eye_trace_aligned[i].cpu()
            reconstructed_cpu = reconstructed[i].cpu()

            # Stitch on CPU
            stitched_cpu = stitch_frames_by_position(
                eye_trace_cpu,
                reconstructed_cpu,
                self.model.img_size
            )

            # Move back to device for loss computation
            stitched = stitched_cpu.to(self.device)
            recon_loss += ((target_img[i] - stitched * mask[i]) ** 2).sum()

        # Compute firing rate constraint loss
        alm_loss, constraint_violation = self.model.compute_firing_rate_loss(rgc_output)

        # Total loss
        total_loss = recon_loss + alm_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Apply constraints
        self.model.normalize_kernels()
        self.model.update_lagrange_multiplier(constraint_violation)

        self.iteration += 1

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar("loss/total", total_loss.item(), self.iteration)
            self.writer.add_scalar("loss/reconstruction", recon_loss.item(), self.iteration)
            self.writer.add_scalar("loss/alm", alm_loss.item(), self.iteration)
            self.writer.add_scalar("constraint/violation_mean", constraint_violation.mean().item(), self.iteration)
            self.writer.add_scalar("constraint/violation_max", constraint_violation.abs().max().item(), self.iteration)

        # Return metrics
        return {
            "iteration": self.iteration,
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "alm_loss": alm_loss.item(),
            "constraint_violation_mean": constraint_violation.mean().item(),
            "constraint_violation_max": constraint_violation.abs().max().item(),
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
            pbar.set_postfix({
                "total": f"{metrics['total_loss']:.2f}",
                "recon": f"{metrics['recon_loss']:.2f}",
                "alm": f"{metrics['alm_loss']:.2f}",
                "constraint": f"{metrics['constraint_violation_mean']:+.4f}",
            })

        if self.writer:
            self.writer.flush()
            self.writer.close()
            print(f"\nTensorboard logs saved to: {self.writer.log_dir}")

        print(f"\nTraining complete! Final iteration: {self.iteration}")
