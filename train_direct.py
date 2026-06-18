"""Training script for DirectReconNet (two-stage: RGC → FrameDecoder)."""

import torch
from datetime import datetime
from src.models.direct_model import DirectReconNet
from src.data.recon_dataset import ReconDataset
from src.training.direct_trainer import DirectTrainer


def main():
    # Model hyperparameters
    img_size = 64
    roi_size = 16
    fix_length = 64

    rgc_channels = 256
    rgc_temporal_length = 16
    rgc_delay = 1  # trailing zero samples in RGC temporal kernel

    # Dataset parameters
    diffusion_coefficient = 10.0 / 3600.0
    sampling_frequency = 360
    pixels_per_degree = 240
    pad_start = rgc_temporal_length - 1  # single convolution stage

    # Constraint parameters
    target_firing_rate = 5.0
    rho = 0

    # Loss weights
    w_recon = 1.0
    w_kernel_var = 1e-4
    w_temporal_smoothness = 1e-4
    w_decorr = 0

    # Reconstruction loss mode: "frames" or "stitched"
    recon_mode = "stitched"

    # Decode with pseudoinverse (Φ⁺ = V Σ⁻¹ Uᵀ) instead of plain transpose (Φᵀ)
    pseudoinverse = True

    # Create dataset
    print("Creating dataset...")
    use_pink = False

    dataset = ReconDataset(
        img_size=img_size,
        roi_size=roi_size,
        total_samples=fix_length,
        pad_start=pad_start,
        diffusion_coefficient=diffusion_coefficient,
        sampling_frequency=sampling_frequency,
        pixels_per_degree=pixels_per_degree,
        use_pink=use_pink,
    )

    # Create model
    print("Creating model...")
    model = DirectReconNet(
        img_size=img_size,
        roi_size=roi_size,
        rgc_channels=rgc_channels,
        rgc_temporal_length=rgc_temporal_length,
        rgc_delay=rgc_delay,
        target_firing_rate=target_firing_rate,
        rho=rho,
    )

    print(f"\nModel: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/direct/{timestamp}"

    trainer = DirectTrainer(
        model=model,
        dataset=dataset,
        batch_size=16,
        learning_rate=2e-3,
        w_recon=w_recon,
        w_kernel_var=w_kernel_var,
        w_temporal_smoothness=w_temporal_smoothness,
        w_decorr=w_decorr,
        recon_mode=recon_mode,
        pseudoinverse=pseudoinverse,
        device=device,
        log_dir=log_dir,
        save_every=100_000,
        checkpoint_dir=f"{log_dir}/checkpoints",
    )

    # Train
    print("\n" + "=" * 80)
    trainer.train(max_iterations=1_000_000)
    print("=" * 80)


if __name__ == "__main__":
    main()
