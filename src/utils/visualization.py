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
