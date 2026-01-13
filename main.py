import torch
from models.retinal_encoder import RetinalEncoder
from models.v1_decoder import V1Decoder
from models.frame_decoder import FrameDecoder
from data.recon_dataset import ReconDataset
from utils.reconstruction import stitch_frames_by_position

def main():
    img_size = 128
    roi_w = 10
    fix_l = 100
    rgc_l = 12
    rgc_n = 100
    v1_l = 12
    v1_n = 1024
    d = 20.0 / 3600.0
    fs = 360
    ppd = 240
    pad_start = (rgc_l - 1) + (v1_l - 1)

    # Firing rate constraint parameters for RGC encoder
    target_firing_rate = 1.0
    rho = 1.0  # Penalty parameter

    dataset = ReconDataset(img_size, roi_w, fix_l, pad_start, d, fs, ppd)

    batch_size = 8

    data_loader = torch.utils.data.DataLoader(dataset, batch_size, False)

    # Initialize RGC encoder with firing rate constraint
    rgc_encoder = RetinalEncoder(
        rgc_n, roi_w, rgc_l,
        target_firing_rate=target_firing_rate,
        rho=rho,
    )
    v1_decoder = V1Decoder(rgc_n, v1_n, v1_l)
    frame_decoder = FrameDecoder(v1_n, roi_w * roi_w)

    # Forward pass
    frames, target_img, eye_trace, mask, _ = next(iter(data_loader))
    print(f"frames: {frames.shape}")

    # RGC encoding with firing rate constraint
    y_rgc = rgc_encoder(frames)
    print(f"RGC output: {y_rgc.shape}")

    # Compute RGC firing rate constraint loss
    rgc_alm_loss, rgc_constraint_violation = rgc_encoder.compute_firing_rate_constraint(y_rgc)

    y_v1 = v1_decoder(y_rgc)
    print(f"V1 output: {y_v1.shape}")
    y = frame_decoder(y_v1)
    y = y.reshape([y.shape[0], -1, roi_w, roi_w])
    print(f"frame decoder output: {y.shape}")

    # Eye trace alignment
    eye_trace = eye_trace[:,:,pad_start:]
    print(f"eye_trace: {eye_trace.shape}")

    # Image reconstruction (first batch item)
    reconstructed = stitch_frames_by_position(eye_trace[0], y[0], img_size)
    print(f"reconstructed: {reconstructed.shape}")

    # Reconstruction loss
    recon_loss = (target_img[0] - reconstructed * mask[0]).square().sum()
    print(f"reconstruction loss: {recon_loss}")

    # Total loss
    total_loss = recon_loss + rgc_alm_loss
    print(f"\nRGC Firing Rate Constraint:")
    print(f"  target firing rate: {target_firing_rate}")
    print(f"  actual firing rate (per channel): min={rgc_constraint_violation.min().item() + target_firing_rate:.6f}, "
          f"mean={rgc_constraint_violation.mean().item() + target_firing_rate:.6f}, "
          f"max={rgc_constraint_violation.max().item() + target_firing_rate:.6f}")
    print(f"  constraint violation (per channel): min={rgc_constraint_violation.min().item():.6f}, "
          f"mean={rgc_constraint_violation.mean().item():.6f}, "
          f"max={rgc_constraint_violation.max().item():.6f}")
    print(f"  Lagrange multiplier: min={rgc_encoder.Lambda.min().item():.6f}, "
          f"mean={rgc_encoder.Lambda.mean().item():.6f}, "
          f"max={rgc_encoder.Lambda.max().item():.6f}")
    print(f"  ALM loss: {rgc_alm_loss.item():.6f}")
    print(f"  total loss: {total_loss.item():.6f}")

    # Update Lagrange multipliers (dual-ascent step for ALM)
    rgc_encoder.update_lagrange_multiplier(rgc_constraint_violation)
    print(f"\nUpdated Lagrange multiplier: min={rgc_encoder.Lambda.min().item():.6f}, "
          f"mean={rgc_encoder.Lambda.mean().item():.6f}, "
          f"max={rgc_encoder.Lambda.max().item():.6f}")

    # Unit-norm kernels
    rgc_encoder.normalize_kenels()


if __name__ == "__main__":
    main()
