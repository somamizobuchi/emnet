"""Example usage of EyeMovementNet - simplified model interface."""

import torch
from src.models import EyeMovementNet
from src.data.recon_dataset import ReconDataset
from src.utils.reconstruction import stitch_frames_by_position


def main():
    # Configuration
    img_size = 128
    roi_size = 10
    fix_l = 100
    rgc_channels = 100
    rgc_temporal = 12
    v1_channels = 1024
    v1_temporal = 12
    d = 20.0 / 3600.0
    fs = 360
    ppd = 240

    # Initialize dataset
    pad_start = (rgc_temporal - 1) + (v1_temporal - 1)
    dataset = ReconDataset(img_size, roi_size, fix_l, pad_start, d, fs, ppd)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    # Initialize COMBINED model (much simpler!)
    model = EyeMovementNet(
        img_size=img_size,
        roi_size=roi_size,
        rgc_channels=rgc_channels,
        rgc_temporal_length=rgc_temporal,
        v1_channels=v1_channels,
        v1_temporal_length=v1_temporal,
        target_firing_rate=1.0,
        rho=1.0,
    )

    print(f"Model: {model}")
    print(f"Temporal reduction: {model.get_temporal_reduction()} frames")

    # Get batch
    frames, target_img, eye_trace, mask, _ = next(iter(data_loader))
    print(f"\nInput frames: {frames.shape}")

    # Forward pass with intermediate outputs
    reconstructed, rgc_out, v1_out = model(frames, return_intermediates=True)

    print(f"RGC output: {rgc_out.shape}")
    print(f"V1 output: {v1_out.shape}")
    print(f"Reconstructed frames: {reconstructed.shape}")

    # Compute firing rate constraint loss
    rgc_alm_loss, constraint_violation = model.compute_firing_rate_loss(rgc_out)

    print(f"\nFiring Rate Constraint:")
    print(f"  ALM loss: {rgc_alm_loss.item():.6f}")
    print(
        f"  Constraint violation: min={constraint_violation.min().item():.6f}, "
        f"mean={constraint_violation.mean().item():.6f}, "
        f"max={constraint_violation.max().item():.6f}"
    )

    # Image reconstruction using first sample
    eye_trace = eye_trace[:, :, pad_start:]
    reconstructed_img = stitch_frames_by_position(
        eye_trace[0], reconstructed[0], img_size
    )

    print(f"\nReconstructed image: {reconstructed_img.shape}")

    # Reconstruction loss
    recon_loss = (target_img[0] - reconstructed_img * mask[0]).square().sum()

    # Compute L2 regularization loss
    l2_loss = model.compute_l2_loss()

    total_loss = recon_loss + rgc_alm_loss

    print(f"Reconstruction loss: {recon_loss.item():.2f}")
    print(f"L2 regularization loss: {l2_loss.item():.2f}")
    print(f"Total loss: {total_loss.item():.2f}")

    # Update constraints and normalize
    model.update_lagrange_multiplier(constraint_violation)
    model.normalize_kernels()

    print(
        f"\nLagrange multipliers updated: min={model.Lambda.min().item():.6f}, "
        f"mean={model.Lambda.mean().item():.6f}, "
        f"max={model.Lambda.max().item():.6f}"
    )


if __name__ == "__main__":
    main()
