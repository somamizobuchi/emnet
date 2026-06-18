"""Simplified symmetric model: RGC encoder + transposed spatial decoder."""

import torch
import torch.nn as nn
from typing import Tuple

from .retinal_encoder import RetinalEncoder


class SimpleReconNet(nn.Module):
    """
    Symmetric encoder–decoder for frame reconstruction.

    Architecture:
        Input (video frames) → Φ (RetinalEncoder) → nonlinearity → Φᵀ → Output

    Stripped of ALM firing-rate constraint and pseudoinverse decoding.
    """

    def __init__(
        self,
        img_size: int,
        roi_size: int,
        rgc_channels: int,
        rgc_temporal_length: int,
        rgc_delay: int = 0,
        noise_std: float = 0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.roi_size = roi_size
        self.rgc_channels = rgc_channels
        self.rgc_temporal_length = rgc_temporal_length
        self.pad_start = rgc_temporal_length - 1
        self.noise_std = noise_std

        self.rgc_encoder = RetinalEncoder(
            n_channels=rgc_channels,
            n_spatial=roi_size,
            n_temporal=rgc_temporal_length,
            delay=rgc_delay,
            target_firing_rate=1.0,
            rho=0.0,
        )

    def forward(
        self,
        frames: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            frames: (batch, time, roi_size, roi_size)
            return_intermediates: if True also return rgc_output

        Returns:
            reconstructed: (batch, reduced_time, roi_size, roi_size)
            [rgc_output: (batch, rgc_channels, reduced_time)]  # only if return_intermediates
        """
        if self.training and self.noise_std > 0.0:
            frames = frames + torch.randn_like(frames) * self.noise_std

        rgc_output = self.rgc_encoder(frames)  # (batch, N, reduced_time)

        W = self.rgc_encoder.spatial_projection.weight  # (N, X²)
        r = rgc_output.transpose(1, 2)                  # (batch, reduced_time, N)
        decoded = r @ W                                  # (batch, reduced_time, X²)

        batch_size, time, _ = decoded.shape
        reconstructed = decoded.reshape(batch_size, time, self.roi_size, self.roi_size)

        if return_intermediates:
            return reconstructed, rgc_output
        return reconstructed

    def compute_sparsity_loss(self, rgc_output: torch.Tensor) -> torch.Tensor:
        """λ₁ ‖z‖₁  — mean L1 norm of RGC activations."""
        return rgc_output.abs().mean()

    def compute_temporal_smoothness_loss(self) -> torch.Tensor:
        """λ₂ Σ_k (h_{k+1} - h_k)²  — first-difference penalty on temporal taps."""
        kernel = self.rgc_encoder._full_kernel().squeeze(1)  # (N, T)
        diff = kernel[:, 1:] - kernel[:, :-1]
        return (diff ** 2).mean()

    def compute_temporal_dc_loss(self) -> torch.Tensor:
        """λ₃ (Σ_k h_k)²  — squared sum of temporal taps penalises DC component."""
        kernel = self.rgc_encoder._full_kernel().squeeze(1)  # (N, T)
        return (kernel.sum(dim=1) ** 2).mean()

    def compute_spatial_variance_loss(self) -> torch.Tensor:
        """Spatial localization penalty (unchanged from direct_model)."""
        return self.rgc_encoder.compute_spatial_variance()

    def normalize_kernels(self) -> None:
        """Normalize spatial and temporal kernels to unit L2 norm per channel."""
        self.rgc_encoder.normalize_kernels()

    def get_temporal_reduction(self) -> int:
        return self.pad_start

    def get_rgc_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rgc_encoder.spatial_weights, self.rgc_encoder.temporal_weights

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, roi_size={self.roi_size}, "
            f"rgc_channels={self.rgc_channels}, "
            f"temporal_reduction={self.pad_start}"
        )
