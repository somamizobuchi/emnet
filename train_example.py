"""Example training script for EyeMovementNet."""

import torch
from datetime import datetime
from src.models.model import EyeMovementNet
from src.data.recon_dataset import ReconDataset
from src.training.trainer import Trainer


def main():
    # Model hyperparameters
    img_size = 64
    roi_size = 20
    fix_length = 128

    rgc_channels = 100
    rgc_temporal_length = 16
    rgc_delay = 1  # trailing zero samples in RGC temporal kernel

    v1_channels = 400
    v1_temporal_length = 16
    v1_delay = 1  # trailing zero samples in V1 temporal kernel

    # Dataset parameters
    diffusion_coefficient = 20.0 / 3600.0
    sampling_frequency = 360
    pixels_per_degree = 240
    pad_start = (rgc_temporal_length - 1) + (v1_temporal_length - 1)

    # Constraint parameters
    target_firing_rate = 1
    rho = 0

    # GradNorm parameters
    grad_norm_alpha = 2

    # Create dataset
    print("Creating dataset...")
    dataset = ReconDataset(
        img_size=img_size,
        roi_size=roi_size,
        total_samples=fix_length,
        pad_start=pad_start,
        diffusion_coefficient=diffusion_coefficient,
        sampling_frequency=sampling_frequency,
        pixels_per_degree=pixels_per_degree,
    )

    # Create model
    print("Creating model...")
    model = EyeMovementNet(
        img_size=img_size,
        roi_size=roi_size,
        rgc_channels=rgc_channels,
        rgc_temporal_length=rgc_temporal_length,
        v1_channels=v1_channels,
        v1_temporal_length=v1_temporal_length,
        rgc_delay=rgc_delay,
        v1_delay=v1_delay,
        target_firing_rate=target_firing_rate,
        rho=rho,
    )

    print(f"\nModel: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/training/{timestamp}"

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=8,
        learning_rate=1e-3,
        grad_norm_alpha=grad_norm_alpha,
        device=device,
        log_dir=log_dir,
        save_every=10_000,
        checkpoint_dir=f"{log_dir}/checkpoints",
    )

    # Train
    print("\n" + "=" * 80)
    trainer.train(max_iterations=1_000_000)
    print("=" * 80)


if __name__ == "__main__":
    main()
