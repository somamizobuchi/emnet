"""Two-stage eye movement model: RetinalEncoder → FrameDecoder (no V1)."""

import torch
import torch.nn as nn
from typing import Tuple

from .retinal_encoder import RetinalEncoder
from .frame_decoder import FrameDecoder


class DirectReconNet(nn.Module):
    """
    Two-stage neural network for eye movement-conditioned image reconstruction.

    Architecture:
        Input (video frames) → RetinalEncoder → FrameDecoder → Output (reconstructed frames)

    Skips the V1Decoder stage of EyeMovementNet to study whether the intermediate
    V1 representation is necessary for reconstruction quality.
    """

    def __init__(
        self,
        img_size: int,
        roi_size: int,
        rgc_channels: int,
        rgc_temporal_length: int,
        rgc_delay: int = 0,
        target_firing_rate: float = 1.0,
        rho: float = 1.0,
    ):
        """
        Initialize the direct reconstruction network.

        Args:
            img_size (int): Size of full image (img_size x img_size)
            roi_size (int): Size of region-of-interest patches (roi_size x roi_size)
            rgc_channels (int): Number of retinal ganglion cell channels
            rgc_temporal_length (int): Total temporal kernel length for RGC encoder (delay + trainable)
            rgc_delay (int): Trailing zero samples in the RGC temporal kernel (default 0)
            target_firing_rate (float): Target firing rate for RGC constraint (default 1.0)
            rho (float): Penalty parameter for Augmented Lagrangian Method (default 1.0)
        """
        super().__init__()

        self.img_size = img_size
        self.roi_size = roi_size
        self.rgc_channels = rgc_channels
        self.rgc_temporal_length = rgc_temporal_length

        # Temporal reduction: only one convolution stage
        self.pad_start = rgc_temporal_length - 1

        self.rgc_encoder = RetinalEncoder(
            n_channels=rgc_channels,
            n_spatial=roi_size,
            n_temporal=rgc_temporal_length,
            delay=rgc_delay,
            target_firing_rate=target_firing_rate,
            rho=rho,
        )

        self.frame_decoder = FrameDecoder(
            in_channels=rgc_channels,
            out_channels=roi_size * roi_size,
        )

    def forward(
        self,
        frames: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the two-stage network.

        Args:
            frames (torch.Tensor): Input video frames, shape (batch, time, roi_size, roi_size)
            return_intermediates (bool): If True, return intermediate activations (default False)

        Returns:
            If return_intermediates=False:
                torch.Tensor: Reconstructed frames, shape (batch, reduced_time, roi_size, roi_size)

            If return_intermediates=True:
                Tuple containing:
                - reconstructed frames (batch, reduced_time, roi_size, roi_size)
                - rgc_output (batch, rgc_channels, time - rgc_temporal + 1)
        """
        # Stage 1: Retinal encoding
        rgc_output = self.rgc_encoder(frames)  # (batch, rgc_channels, reduced_time)

        # Stage 2: Frame decoding directly from RGC
        decoded = self.frame_decoder(rgc_output)  # (batch, reduced_time, roi_size²)

        batch_size, time, _ = decoded.shape
        reconstructed = decoded.reshape(batch_size, time, self.roi_size, self.roi_size)

        if return_intermediates:
            return reconstructed, rgc_output
        else:
            return reconstructed

    def compute_firing_rate_loss(
        self,
        rgc_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute firing rate constraint loss for RGC layer.

        Args:
            rgc_output (torch.Tensor): RGC activations, shape (batch, n_channels, time)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - alm_loss: Augmented Lagrangian loss (scalar)
                - constraint_violation: Constraint violation per channel, shape (n_channels,)
        """
        return self.rgc_encoder.compute_firing_rate_constraint(rgc_output)

    def compute_spatial_variance_loss(self) -> torch.Tensor:
        """
        Compute the spatial variance (localization) loss for the RGC encoder kernels.

        Returns:
            torch.Tensor: Mean spatial variance (scalar)
        """
        return self.rgc_encoder.compute_spatial_variance()

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss: L2 for frame decoder weights.

        Returns:
            torch.Tensor: Regularization loss (scalar)
        """
        return (self.frame_decoder.decoder.weight ** 2).mean()

    def compute_temporal_smoothness_loss(self) -> torch.Tensor:
        """Second-derivative smoothness penalty on RGC temporal taps."""
        return self.rgc_encoder.compute_temporal_smoothness_loss()

    def update_lagrange_multiplier(self, constraint_violation: torch.Tensor) -> None:
        """
        Update Lagrange multipliers for firing rate constraint (dual-ascent step).

        Args:
            constraint_violation (torch.Tensor): Constraint violation per channel, shape (n_channels,)
        """
        self.rgc_encoder.update_lagrange_multiplier(constraint_violation)

    def normalize_kernels(self) -> None:
        """Normalize RGC filters to unit L2 norm (energy constraint)."""
        self.rgc_encoder.normalize_kernels()

    def get_temporal_reduction(self) -> int:
        """
        Get total temporal reduction through the network.

        Returns:
            int: Number of frames reduced from input to output
        """
        return self.pad_start

    @property
    def Lambda(self) -> torch.Tensor:
        """Access Lagrange multipliers for firing rate constraint."""
        val = self.rgc_encoder.lagrange_multiplier
        assert isinstance(val, torch.Tensor)
        return val

    def get_rgc_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get spatial and temporal weights from the RGC encoder.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - spatial_weights: (N, X, X)
                - temporal_weights: (N, T)
        """
        return self.rgc_encoder.spatial_weights, self.rgc_encoder.temporal_weights

    def extra_repr(self) -> str:
        """Extra representation for printing the model."""
        return (
            f"img_size={self.img_size}, roi_size={self.roi_size}, "
            f"rgc_channels={self.rgc_channels}, "
            f"temporal_reduction={self.pad_start}"
        )
