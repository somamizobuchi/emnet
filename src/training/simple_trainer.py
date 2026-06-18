"""Iteration-based trainer for SimpleReconNet (frame reconstruction only)."""

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from pathlib import Path
from tqdm import tqdm

from ..models.simple_model import SimpleReconNet
import torch.nn.functional as F

from ..utils import visualization as viz


class SimpleTrainer:
    """
    Iteration-based trainer for SimpleReconNet.

    Loss = w_recon * L1(reconstructed, target_patches)
         + w_sparsity    * mean(|z|)                 [λ₁ ‖z‖₁]
         + w_smoothness  * mean((h_{k+1} - h_k)²)   [λ₂ Σ first-differences²]
         + w_dc          * mean((Σ h_k)²)            [λ₃ (Σ h_k)²]
    """

    def __init__(
        self,
        model: SimpleReconNet,
        dataset: Dataset,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        w_recon: float = 1.0,
        w_sparsity: float = 1e-3,
        w_smoothness: float = 1e-3,
        w_dc: float = 1e-3,
        w_kernel_var: float = 1e-3,
        recon_loss: str = "l2",
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
        self.w_sparsity = w_sparsity
        self.w_smoothness = w_smoothness
        self.w_dc = w_dc
        self.w_kernel_var = w_kernel_var
        if recon_loss not in ("l1", "l2"):
            raise ValueError(f"recon_loss must be 'l1' or 'l2', got {recon_loss!r}")
        self.recon_loss = recon_loss

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.data_iter = iter(self.dataloader)

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

        self.writer = SummaryWriter(log_dir) if log_dir else None

        self.save_every = save_every
        if save_every is not None:
            self.checkpoint_dir = Path(checkpoint_dir or log_dir or "checkpoints")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.iteration = 0

    def save_checkpoint(self, path: Optional[str] = None):
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

    def _get_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        frames, target_img, eye_trace, _, _ = batch
        frames = frames.to(self.device)
        target_img = target_img.to(self.device)
        return frames, target_img

    def train_step(self) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        frames, _ = self._get_batch()

        reconstructed, rgc_output = self.model(frames, return_intermediates=True)

        pad_start = self.model.get_temporal_reduction()
        target_patches = frames[:, pad_start:, :, :]

        recon_loss = (
            F.mse_loss(reconstructed, target_patches)
            if self.recon_loss == "l2"
            else F.l1_loss(reconstructed, target_patches)
        )
        sparsity_loss = self.model.compute_sparsity_loss(rgc_output)
        smoothness_loss = self.model.compute_temporal_smoothness_loss()
        dc_loss = self.model.compute_temporal_dc_loss()
        kernel_var_loss = self.model.compute_spatial_variance_loss()

        total_loss = (
            self.w_recon * recon_loss
            + self.w_sparsity * sparsity_loss
            + self.w_smoothness * smoothness_loss
            + self.w_dc * dc_loss
            + self.w_kernel_var * kernel_var_loss
        )

        total_loss.backward()
        self.optimizer.step()

        self.model.normalize_kernels()

        self.iteration += 1

        if self.writer and self.iteration % 100 == 0:
            self.writer.add_scalar("loss/total", float(total_loss.item()), self.iteration)
            self.writer.add_scalar("loss/reconstruction", float(recon_loss.item()), self.iteration)
            self.writer.add_scalar("loss/sparsity", float(sparsity_loss.item()), self.iteration)
            self.writer.add_scalar("loss/smoothness", float(smoothness_loss.item()), self.iteration)
            self.writer.add_scalar("loss/dc", float(dc_loss.item()), self.iteration)
            self.writer.add_scalar("loss/kernel_variance", float(kernel_var_loss.item()), self.iteration)

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

            with torch.no_grad():
                recon_frames = reconstructed[-1].detach().cpu()    # (T, H, W)
                target_frames = target_patches[-1].detach().cpu()  # (T, H, W)
                lo = min(recon_frames.min(), target_frames.min())
                hi = max(recon_frames.max(), target_frames.max())
                recon_norm = (recon_frames - lo) / (hi - lo + 1e-8)    # (T, H, W)
                target_norm = (target_frames - lo) / (hi - lo + 1e-8)  # (T, H, W)
                # Log middle frame as a side-by-side image (avoids broken add_video)
                mid = recon_norm.shape[0] // 2
                side_by_side = torch.cat([recon_norm[mid], target_norm[mid]], dim=1)  # (H, 2W)
                self.writer.add_image(
                    "recon/reconstructed_vs_target",
                    side_by_side.unsqueeze(0),
                    self.iteration,
                )

        return {
            "iteration": self.iteration,
            "total_loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "sparsity_loss": float(sparsity_loss.item()),
            "smoothness_loss": float(smoothness_loss.item()),
            "dc_loss": float(dc_loss.item()),
            "kernel_var_loss": float(kernel_var_loss.item()),
        }

    def train(self, max_iterations: int):
        print(f"Starting training for {max_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(
            f"Loss weights: recon={self.w_recon}, sparsity={self.w_sparsity}, "
            f"smoothness={self.w_smoothness}, dc={self.w_dc}"
        )
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
                }
            )

        if self.writer:
            self.writer.flush()
            self.writer.close()
            print(f"\nTensorboard logs saved to: {self.writer.log_dir}")

        print(f"\nTraining complete! Final iteration: {self.iteration}")
