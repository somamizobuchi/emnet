"""
Visualise RGC spatial + temporal kernel pairs and their radial power spectra.

For each RGC channel (up to --max-neurons) the script produces:
  • A grid of (spatial kernel image | temporal kernel waveform) pairs.
  • A log-log plot of the radial-average power spectrum of each spatial kernel,
    with the mean spectrum overlaid in bold.

Usage
-----
    python analyze_rgc_kernels.py <checkpoint.pt> [--out DIR] [--max-neurons N] [--ncols N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_weights(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (spatial, temporal, iteration) from a checkpoint.

    spatial  : (N, X, X)   float32
    temporal : (N, T)      float32  — full kernel including delay zeros
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    iteration = ckpt.get("iteration", 0)
    state = ckpt["model_state_dict"]

    # Spatial weights: (N, X*X) → (N, X, X)
    spatial_flat = state["rgc_encoder.spatial_projection.weight"].float()
    N = spatial_flat.shape[0]
    X = int(round(spatial_flat.shape[1] ** 0.5))
    spatial = spatial_flat.view(N, X, X).numpy()

    # Temporal weights: (N, 1, T_train); delay zeros appended if present
    temporal_raw = state["rgc_encoder.temporal_weight"].float().squeeze(1)  # (N, T_train)

    # Reconstruct delay from any stored buffer or infer from checkpoint keys
    # We just use the trainable taps as-is (delay zeros not in temporal_weight)
    temporal = temporal_raw.numpy()  # (N, T_train)

    return spatial, temporal, int(iteration)


def _radial_power_spectrum(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial-average power spectrum of a 2-D spatial kernel.

    Returns
    -------
    freq_bins : 1-D array of radial spatial frequencies (cycles / pixel)
    power     : 1-D array of mean power at each frequency bin
    """
    X = kernel.shape[0]
    # 2-D FFT, shift DC to centre
    f2d = np.fft.fftshift(np.fft.fft2(kernel))
    power2d = np.abs(f2d) ** 2

    # Radial frequency map (cycles / pixel)
    fx = np.fft.fftshift(np.fft.fftfreq(X))  # [-0.5, 0.5)
    fy = np.fft.fftshift(np.fft.fftfreq(X))
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    R = np.sqrt(FX ** 2 + FY ** 2)

    # Bin by radial frequency; use X//2 bins up to Nyquist (0.5 c/px)
    n_bins = X // 2
    bin_edges = np.linspace(0.0, 0.5, n_bins + 1)
    freq_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (R >= bin_edges[i]) & (R < bin_edges[i + 1])
        if mask.any():
            power[i] = power2d[mask].mean()

    return freq_bins, power


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_kernel_pairs(
    spatial: np.ndarray,
    temporal: np.ndarray,
    out_dir: Path,
    max_neurons: int = 64,
    ncols: int = 8,
    iteration: int = 0,
) -> None:
    """Grid of (spatial image | temporal waveform) for each RGC neuron."""
    N = min(spatial.shape[0], max_neurons)
    n_pairs = ncols  # pairs per row
    nrows = (N + n_pairs - 1) // n_pairs

    # Each neuron occupies 2 sub-columns: [spatial | temporal]
    fig, axes = plt.subplots(
        nrows, n_pairs * 2,
        figsize=(n_pairs * 2 * 1.4, nrows * 1.8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.6},
    )
    axes = np.array(axes).reshape(nrows, n_pairs * 2)

    vmax = np.abs(spatial[:N]).max()
    vmax = vmax if vmax > 0 else 1.0

    T = temporal.shape[1]
    t_axis = np.arange(T)

    for idx in range(nrows * n_pairs):
        row = idx // n_pairs
        col_pair = idx % n_pairs
        ax_s = axes[row, col_pair * 2]      # spatial
        ax_t = axes[row, col_pair * 2 + 1]  # temporal

        ax_s.axis("off")
        ax_t.axis("off")

        if idx >= N:
            continue

        ax_s.imshow(
            spatial[idx],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_s.set_title(f"RGC {idx}", fontsize=5, pad=1)
        ax_s.axis("on")
        ax_s.set_xticks([])
        ax_s.set_yticks([])

        ax_t.plot(t_axis, temporal[idx], lw=0.7, color="steelblue")
        ax_t.axhline(0, color="k", lw=0.4, ls="--")
        ax_t.axis("on")
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        ax_t.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"RGC kernel pairs  (iter {iteration}, N={N})\n"
        "Left: spatial  |  Right: temporal",
        fontsize=9,
    )
    path = out_dir / "rgc_kernel_pairs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_power_spectra(
    spatial: np.ndarray,
    out_dir: Path,
    max_neurons: int = 256,
    iteration: int = 0,
) -> None:
    """Log-log radial power spectra for each RGC spatial kernel + mean."""
    N = min(spatial.shape[0], max_neurons)

    all_power = []
    freq_bins = None

    for i in range(N):
        fb, pw = _radial_power_spectrum(spatial[i])
        if freq_bins is None:
            freq_bins = fb
        all_power.append(pw)

    all_power = np.stack(all_power)  # (N, n_bins)
    mean_power = all_power.mean(axis=0)

    # Keep only bins with non-zero frequency and positive power for log-log
    valid = (freq_bins > 0) & (mean_power > 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(N):
        pw = all_power[i]
        v = (freq_bins > 0) & (pw > 0)
        ax.plot(freq_bins[v], pw[v], lw=0.4, alpha=0.25, color="steelblue")

    ax.plot(
        freq_bins[valid],
        mean_power[valid],
        lw=2.0,
        color="crimson",
        label=f"Mean (N={N})",
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spatial frequency (cycles / pixel)", fontsize=10)
    ax.set_ylabel("Power", fontsize=10)
    ax.set_title(
        f"Radial-average power spectrum of RGC spatial kernels\n(iter {iteration})",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()

    path = out_dir / "rgc_spatial_power_spectra.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: same folder as checkpoint)",
    )
    parser.add_argument(
        "--max-neurons", type=int, default=64,
        help="Max RGC neurons to show in the kernel-pair grid (default 64)",
    )
    parser.add_argument(
        "--ncols", type=int, default=8,
        help="Number of neuron columns in the kernel-pair grid (default 8)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    out_dir = args.out or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    spatial, temporal, iteration = _load_weights(args.checkpoint)
    N, X, _ = spatial.shape
    T = temporal.shape[1]
    print(f"  Iteration : {iteration}")
    print(f"  RGC channels : {N},  roi_size : {X},  temporal taps : {T}")

    print("Plotting kernel pairs...")
    plot_kernel_pairs(
        spatial, temporal, out_dir,
        max_neurons=args.max_neurons,
        ncols=args.ncols,
        iteration=iteration,
    )

    print("Plotting power spectra...")
    plot_power_spectra(spatial, out_dir, max_neurons=N, iteration=iteration)

    print("\nDone.")


if __name__ == "__main__":
    main()
