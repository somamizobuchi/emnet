"""Training script for SimpleReconNet (frame reconstruction only)."""

import torch
from datetime import datetime
from src.models.simple_model import SimpleReconNet
from src.data.recon_dataset import ReconDataset
from src.data.stpink_dataset import STPinkDataset
from src.training.simple_trainer import SimpleTrainer


def main():
    # Model hyperparameters
    img_size = 64
    roi_size = 20
    fix_length = 64

    rgc_channels = 128
    rgc_temporal_length = 16
    rgc_delay = 1  # trailing zero samples in RGC temporal kernel

    noise_std = 1

    # Dataset parameters
    diffusion_coefficient = 10.0 / 3600.0
    sampling_frequency = 360
    pixels_per_degree = 240
    pad_start = rgc_temporal_length - 1  # single convolution stage

    # Loss weights
    w_recon = 1.0
    w_sparsity = 1e-3  # λ₁: L1 penalty on RGC activations z
    w_smoothness = 1e-2  # λ₂: first-difference smoothness on temporal taps h
    w_dc = 1e-2  # λ₃: squared-sum penalty on temporal taps h
    w_kernel_var = 1e-4  # spatial localization penalty on RGC kernels

    # Reconstruction loss: "l1" or "l2"
    recon_loss = "l1"

    # Input mode: "eye" (eye-movement modulated patches) or
    # "stpink" (spatiotemporal pink noise control)
    input_mode = "eye"
    stpink_alpha = 1.0

    # Create dataset
    print("Creating dataset...")
    if input_mode == "stpink":
        dataset = STPinkDataset(
            roi_size=roi_size,
            total_samples=fix_length,
            alpha=stpink_alpha,
        )
    elif input_mode == "eye":
        dataset = ReconDataset(
            img_size=img_size,
            roi_size=roi_size,
            total_samples=fix_length,
            pad_start=pad_start,
            diffusion_coefficient=diffusion_coefficient,
            sampling_frequency=sampling_frequency,
            pixels_per_degree=pixels_per_degree,
        )
    else:
        raise ValueError(f"Unknown input_mode: {input_mode!r}")

    # Create model
    print("Creating model...")
    model = SimpleReconNet(
        img_size=img_size,
        roi_size=roi_size,
        rgc_channels=rgc_channels,
        rgc_temporal_length=rgc_temporal_length,
        rgc_delay=rgc_delay,
        noise_std=noise_std,
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
    log_dir = f"runs/simple/{timestamp}"

    trainer = SimpleTrainer(
        model=model,
        dataset=dataset,
        batch_size=16,
        learning_rate=2e-3,
        w_recon=w_recon,
        w_sparsity=w_sparsity,
        w_smoothness=w_smoothness,
        w_dc=w_dc,
        w_kernel_var=w_kernel_var,
        recon_loss=recon_loss,
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
