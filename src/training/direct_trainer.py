"""Iteration-based trainer for DirectReconNet."""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from pathlib import Path
from tqdm import tqdm

from ..models.direct_model import DirectReconNet
from ..data.recon_dataset import ReconDataset
import torch.nn.functional as F

from ..utils.reconstruction import stitch_frames_by_position
from ..utils import visualization as viz


class DirectTrainer:
    """
    Iteration-based trainer for DirectReconNet (two-stage: RGC → FrameDecoder).

    Loss = w_recon * L1_recon
         + w_kernel_var * kernel_variance
         + w_reg * L2_regularization
         + w_temporal_smoothness * temporal_smoothness

    recon_mode:
        "frames"   — L1 loss between reconstructed and target patches directly
        "stitched" — L1 loss between images stitched from reconstructed/target
                     patches using eye positions (via stitch_frames_by_position)
    """

    def __init__(
        self,
        model: DirectReconNet,
        dataset: ReconDataset,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        w_recon: float = 1.0,
        w_kernel_var: float = 1e-3,
        w_reg: float = 1e-4,
        w_temporal_smoothness: float = 1e-3,
        w_decorr: float = 0.0,
        recon_mode: str = "frames",
        device: str = "cpu",
        log_dir: Optional[str] = None,
        save_every: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.w_recon = w_recon
        self.w_kernel_var = w_kernel_var
        self.w_reg = w_reg
        self.w_temporal_smoothness = w_temporal_smoothness
        self.w_decorr = w_decorr

        if recon_mode not in ("frames", "stitched"):
            raise ValueError(f"recon_mode must be 'frames' or 'stitched', got {recon_mode!r}")
        self.recon_mode = recon_mode

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
                "rgc_delay": self.model.rgc_encoder.D,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    @staticmethod
    def _decorrelation_loss(target_patches: torch.Tensor) -> torch.Tensor:
        """Penalize cross-correlation between spatial pixels over time.

        target_patches: (B, T, H, W)
        Reshapes to (B, N, T), computes gram matrix (B, N, N), returns mean
        squared off-diagonal — encouraging decorrelated pixel responses.
        """
        B, T, H, W = target_patches.shape
        N = H * W
        x = target_patches.view(B, N, T)          # (B, N, T)
        C = x @ x.transpose(-1, -2)               # (B, N, N)
        mask = ~torch.eye(N, dtype=torch.bool, device=target_patches.device)
        return C[:, mask].square().mean()

    def _get_batch(self):
        """Get next batch, restarting iterator if needed."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        frames, target_img, eye_trace, _, _ = batch
        frames = frames.to(self.device)
        target_img = target_img.to(self.device)
        eye_trace = eye_trace.to(self.device)

        return frames, target_img, eye_trace

    def train_step(self) -> dict:
        """Execute one training iteration."""
        self.model.train()
        self.optimizer.zero_grad()

        # Get batch
        frames, target_img, eye_trace = self._get_batch()

        # Forward pass
        reconstructed, rgc_output = self.model(frames, return_intermediates=True)

        # Trim frames to valid convolution length
        pad_start = self.model.get_temporal_reduction()
        target_patches = frames[:, pad_start:, :, :]
        eye_trace_aligned = eye_trace[:, :, pad_start:]

        # Reconstruction loss
        if self.recon_mode == "frames":
            recon_loss = F.l1_loss(reconstructed, target_patches)
        else:  # "stitched"
            recon_imgs = torch.stack([
                stitch_frames_by_position(eye_trace_aligned[b], reconstructed[b], self.model.img_size)
                for b in range(reconstructed.shape[0])
            ])
            target_imgs = torch.stack([
                stitch_frames_by_position(eye_trace_aligned[b], target_patches[b], self.model.img_size)
                for b in range(target_patches.shape[0])
            ])
            recon_loss = F.l1_loss(recon_imgs, target_imgs)
        kernel_var_loss = self.model.compute_spatial_variance_loss()
        reg_loss = self.model.compute_regularization_loss()
        temporal_smoothness_loss = self.model.compute_temporal_smoothness_loss()
        decorr_loss = self._decorrelation_loss(target_patches)
        alm_loss, constraint_violation = self.model.compute_firing_rate_loss(rgc_output)

        total_loss = (
            self.w_recon * recon_loss
            + self.w_kernel_var * kernel_var_loss
            + self.w_reg * reg_loss
            + self.w_temporal_smoothness * temporal_smoothness_loss
            + self.w_decorr * decorr_loss
            + alm_loss
        )

        total_loss.backward()
        self.optimizer.step()

        # Energy constraint: normalize RGC kernels to unit L2 norm
        self.model.normalize_kernels()

        # Dual-ascent step: update Lagrange multipliers for firing rate constraint
        self.model.update_lagrange_multiplier(constraint_violation.detach())

        self.iteration += 1

        # Tensorboard logging
        if self.writer and self.iteration % 100 == 0:
            self.writer.add_scalar("loss/total", float(total_loss.item()), self.iteration)
            self.writer.add_scalar("loss/reconstruction", float(recon_loss.item()), self.iteration)
            self.writer.add_scalar("loss/kernel_variance", float(kernel_var_loss.item()), self.iteration)
            self.writer.add_scalar("loss/regularization", float(reg_loss.item()), self.iteration)
            self.writer.add_scalar("loss/temporal_smoothness", float(temporal_smoothness_loss.item()), self.iteration)
            self.writer.add_scalar("loss/decorrelation", float(decorr_loss.item()), self.iteration)
            self.writer.add_scalar("loss/alm", float(alm_loss.item()), self.iteration)
            self.writer.add_scalar("constraint/firing_rate_violation", float(constraint_violation.abs().mean().item()), self.iteration)

        if self.writer and self.iteration % 1000 == 0:
            spatial, temporal = self.model.get_rgc_weights()

            spatial_grid = viz.plot_spatial_kernels(spatial, self.model.roi_size)
            self.writer.add_image("rgc/spatial_kernels", spatial_grid, self.iteration)

            temporal_plot = viz.plot_temporal_kernels(
                temporal,
                title="RGC Temporal Kernels",
                delay=self.model.rgc_encoder.D,
            )
            self.writer.add_image("rgc/temporal_kernels", temporal_plot, self.iteration)

            log_gain = self.model.rgc_encoder.log_gain
            if torch.isfinite(log_gain).all():
                self.writer.add_histogram("rgc/log_gain", log_gain, self.iteration)

            log_bias = self.model.rgc_encoder.log_bias
            if torch.isfinite(log_bias).all():
                self.writer.add_histogram("rgc/log_bias", log_bias, self.iteration)

            def norm_img(img):
                return (img - img.min()) / (img.max() - img.min() + 1e-8)

            if self.recon_mode == "frames":
                # Log a side-by-side video: (T, 1, H, 2W) — reconstructed | target
                with torch.no_grad():
                    recon_frames = reconstructed[-1].detach().cpu()   # (T, H, W)
                    target_frames = target_patches[-1].detach().cpu() # (T, H, W)
                    # Normalise jointly so brightness is comparable
                    lo = min(recon_frames.min(), target_frames.min())
                    hi = max(recon_frames.max(), target_frames.max())
                    recon_norm = ((recon_frames - lo) / (hi - lo + 1e-8)).unsqueeze(1)  # (T, 1, H, W)
                    target_norm = ((target_frames - lo) / (hi - lo + 1e-8)).unsqueeze(1)
                    side_by_side = torch.cat([recon_norm, target_norm], dim=3).expand(-1, 3, -1, -1)  # (T, 3, H, 2W)
                    # add_video expects (N, T, C, H, W)
                    self.writer.add_video(
                        "recon/reconstructed_vs_target",
                        side_by_side.unsqueeze(0),
                        self.iteration,
                        fps=30,
                    )
            else:
                self.writer.add_image(
                    "recon/target", norm_img(target_img[-1]), self.iteration, dataformats="HW"
                )
                with torch.no_grad():
                    trace_cpu = eye_trace_aligned[-1].cpu()
                    recon_cpu = reconstructed[-1].detach().cpu()
                    stitched = stitch_frames_by_position(
                        trace_cpu, recon_cpu, self.model.img_size
                    )
                self.writer.add_image(
                    "recon/reconstructed", norm_img(stitched), self.iteration, dataformats="HW"
                )

        return {
            "iteration": self.iteration,
            "total_loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "kernel_var_loss": float(kernel_var_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "temporal_smoothness_loss": float(temporal_smoothness_loss.item()),
            "decorr_loss": float(decorr_loss.item()),
            "alm_loss": float(alm_loss.item()),
            "firing_rate_violation": float(constraint_violation.abs().mean().item()),
        }

    def train(self, max_iterations: int):
        """Train for a fixed number of iterations."""
        print(f"Starting training for {max_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Recon mode: {self.recon_mode}")
        print(f"Loss weights: recon={self.w_recon}, kernel_var={self.w_kernel_var}, "
              f"reg={self.w_reg}, temporal_smoothness={self.w_temporal_smoothness}")
        print()

        pbar = tqdm(range(max_iterations), desc="Training")
        for _ in pbar:
            metrics = self.train_step()

            if self.save_every and self.iteration % self.save_every == 0:
                self.save_checkpoint()

            pbar.set_postfix(
                {
                    "total": f"{metrics['total_loss']:.4f}",
                    "recon": f"{metrics['recon_loss']:.4f}",
                    "kvar": f"{metrics['kernel_var_loss']:.4f}",
                    "smooth": f"{metrics['temporal_smoothness_loss']:.4f}",
                }
            )

        if self.writer:
            self.writer.flush()
            self.writer.close()
            print(f"\nTensorboard logs saved to: {self.writer.log_dir}")

        print(f"\nTraining complete! Final iteration: {self.iteration}")
