import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Optional


def plot_temporal_kernels(weights: torch.Tensor, title: str = "Temporal Kernels", delay: int = 0) -> torch.Tensor:
    """
    Plot all temporal kernels overlaid.

    Args:
        weights (torch.Tensor): Temporal weights of shape (N, T), including any delay zeros
        title (str): Plot title
        delay (int): Number of leading zero (delay) samples — drawn as a shaded region

    Returns:
        torch.Tensor: Image tensor of shape (3, H, W)
    """
    weights_np = weights.detach().cpu().numpy()
    fig = plt.figure(figsize=(8, 6))
    plt.plot(weights_np.T, alpha=0.3)
    if delay > 0:
        T = weights_np.shape[1]
        plt.axvspan(T - delay - 0.5, T - 0.5, color="gray", alpha=0.15, label=f"delay ({delay} samples)")
        plt.legend(fontsize=8)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Weight")
    plt.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = plt.imread(buf)
    # Convert to RGB (dropping alpha if present) and move channel to front
    img_rgb = img[:, :, :3].transpose(2, 0, 1)
    return torch.from_numpy(img_rgb)


def plot_spatial_kernels(
    weights: torch.Tensor, kernel_size: int, spacing: int = 1
) -> torch.Tensor:
    """
    Create a grid of spatial kernels.

    Args:
        weights (torch.Tensor): Spatial weights of shape (N, X*X)
        kernel_size (int): Size of each kernel (X)
        spacing (int): Pixels between kernels in the grid

    Returns:
        torch.Tensor: Image tensor of shape (1, H, W) or (3, H, W)
    """
    N = weights.shape[0]
    rows = int(np.ceil(np.sqrt(N)))
    cols = int(np.ceil(N / rows))

    grid_h = rows * (kernel_size + spacing) - spacing
    grid_w = cols * (kernel_size + spacing) - spacing

    # Use a diverging colormap (RdBu_r) for spatial weights (centered at 0)
    # We'll use matplotlib to render the individual kernels if we want color,
    # but for simplicity, let's start with a grayscale normalized grid.
    grid = torch.ones((1, grid_h, grid_w), dtype=torch.float32) * 0.5

    weights_reshaped = weights.detach().cpu().view(N, kernel_size, kernel_size)

    # Global normalisation: map [-vmax, vmax] → [0, 1] so relative magnitudes are preserved
    vmax = weights_reshaped.abs().max()
    if vmax > 0:
        weights_norm = (weights_reshaped / vmax) * 0.5 + 0.5
    else:
        weights_norm = torch.full_like(weights_reshaped, 0.5)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < N:
                x = i * (kernel_size + spacing)
                y = j * (kernel_size + spacing)
                grid[0, x : x + kernel_size, y : y + kernel_size] = weights_norm[idx]

    return grid


def plot_temporal_kernels_freq(
    weights: torch.Tensor,
    sampling_frequency: float = 1.0,
    title: str = "Temporal Kernels — Frequency Domain",
    delay: int = 0,
) -> torch.Tensor:
    """
    Plot the magnitude spectrum of every temporal kernel overlaid.

    Args:
        weights: Shape (N, T). Delay taps should be included (they are trimmed
                 before the FFT so the leading zeros don't bias the spectrum).
        sampling_frequency: Sampling rate in Hz — sets the x-axis scale.
        title: Plot title.
        delay: Number of leading delay taps to strip before the FFT.

    Returns:
        torch.Tensor: Image tensor of shape (3, H, W).
    """
    w = weights.detach().cpu().numpy()
    if delay > 0:
        w = w[:, :-delay]  # remove trailing delay zeros

    T = w.shape[1]
    freqs = np.fft.rfftfreq(T, d=1.0 / sampling_frequency)
    spectra = np.abs(np.fft.rfft(w, axis=1))  # (N, T//2+1)

    mean_spectrum = spectra.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in spectra:
        ax.plot(freqs, row, color="steelblue", alpha=0.15, linewidth=0.8)
    ax.plot(freqs, mean_spectrum, color="navy", linewidth=1.5, label="mean")
    ax.set_xlabel("Frequency (Hz)" if sampling_frequency != 1.0 else "Normalised frequency")
    ax.set_ylabel("|FFT|")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    return torch.from_numpy(img[:, :, :3].transpose(2, 0, 1))


def _radial_average(power2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the radial average of a 2D power spectrum (already fftshifted).

    Returns (bin_centers, mean_power) where bin_centers are in cycles/pixel
    (0 … 0.5).
    """
    H, W = power2d.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    fy = (np.arange(H) - cy) / H  # cycles/pixel, centered
    fx = (np.arange(W) - cx) / W
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    max_r = min(cy / H, cx / W)  # largest full ring
    n_bins = min(H, W) // 2
    bins = np.linspace(0, max_r, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    mean_power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.any():
            mean_power[i] = power2d[mask].mean()

    return centers, mean_power


def plot_spatial_kernels_freq(
    weights: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Plot the radially-averaged power spectrum of the spatial kernels.

    Each kernel contributes one curve (light blue); the mean across kernels is
    overlaid in navy. X-axis is spatial frequency in cycles/pixel (0 … 0.5).

    Returns:
        torch.Tensor: Image tensor of shape (3, H, W).
    """
    N = weights.shape[0]
    w = weights.detach().cpu().view(N, kernel_size, kernel_size).numpy()

    power = np.abs(np.fft.fftshift(np.fft.fft2(w), axes=(-2, -1))) ** 2  # (N, H, W)

    profiles = []
    for i in range(N):
        freqs, profile = _radial_average(power[i])
        profiles.append(profile)
    profiles = np.array(profiles)  # (N, n_bins)
    mean_profile = profiles.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    for p in profiles:
        ax.plot(freqs, p, color="steelblue", alpha=0.15, linewidth=0.8)
    ax.plot(freqs, mean_profile, color="navy", linewidth=1.5, label="mean")
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Power")
    ax.set_title("Spatial kernels — radially averaged power spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    return torch.from_numpy(img[:, :, :3].transpose(2, 0, 1))
