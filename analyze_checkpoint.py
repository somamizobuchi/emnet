"""
Analyze a training checkpoint.

For each V1 neuron, reconstruct a 2-D "connection map" in RGC spatial space:
  - Compute the centroid (center-of-mass of squared weights) for every RGC
    spatial kernel.
  - For V1 neuron i, place a point at the centroid of RGC j with value equal
    to the signed V1 weight W_v1[i, j].  The final image is the sum of all
    such contributions (a weighted Dirac comb smoothed with a Gaussian).

Usage
-----
    python analyze_checkpoint.py <checkpoint.pt> [--sigma SIGMA] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models.model import EyeMovementNet


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def compute_rgc_centroids(spatial_weights: torch.Tensor) -> np.ndarray:
    """
    Compute the centroid of each RGC spatial kernel.

    Parameters
    ----------
    spatial_weights : Tensor shape (N_rgc, X, X)

    Returns
    -------
    centroids : ndarray shape (N_rgc, 2)  — (row, col) in pixel coordinates
    """
    w = spatial_weights.detach().cpu().float()  # (N, X, X)
    mass = w.pow(2)                              # use squared weights as mass
    N, X, _ = mass.shape

    coords = torch.arange(X, dtype=torch.float32)  # [0 .. X-1]

    # Marginal sums
    mass_row = mass.sum(dim=2)   # (N, X)  — summed over columns → row mass
    mass_col = mass.sum(dim=1)   # (N, X)  — summed over rows    → col mass

    total = mass.sum(dim=(1, 2)).clamp(min=1e-12)  # (N,)

    centroid_row = (mass_row * coords.unsqueeze(0)).sum(dim=1) / total  # (N,)
    centroid_col = (mass_col * coords.unsqueeze(0)).sum(dim=1) / total  # (N,)

    return torch.stack([centroid_row, centroid_col], dim=1).numpy()  # (N, 2)


# ---------------------------------------------------------------------------
# V1 connection map
# ---------------------------------------------------------------------------

def build_v1_connection_maps(
    centroids: np.ndarray,
    v1_weights: np.ndarray,
    roi_size: int,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Build a 2-D connection map for every V1 neuron.

    Each RGC centroid is rendered as a Gaussian blob scaled by the signed
    V1 → RGC weight.  Positive weights are red, negative are blue when
    displayed with a diverging colormap.

    Parameters
    ----------
    centroids  : (N_rgc, 2) float — (row, col) centroid positions
    v1_weights : (N_v1, N_rgc) float — V1 spatial projection weights
    roi_size   : int — side length of the output image (pixels)
    sigma      : float — std-dev of the Gaussian spread around each centroid

    Returns
    -------
    maps : ndarray (N_v1, roi_size, roi_size)
    """
    N_v1, N_rgc = v1_weights.shape
    maps = np.zeros((N_v1, roi_size, roi_size), dtype=np.float32)

    # Pre-build Gaussian kernels for each RGC (centroid may be fractional)
    row_grid, col_grid = np.meshgrid(
        np.arange(roi_size, dtype=np.float32),
        np.arange(roi_size, dtype=np.float32),
        indexing="ij",
    )

    for j in range(N_rgc):
        cr, cc = centroids[j]
        blob = np.exp(
            -((row_grid - cr) ** 2 + (col_grid - cc) ** 2) / (2.0 * sigma ** 2)
        )
        # Add this blob, scaled by each V1 neuron's weight for RGC j
        # v1_weights[:, j] has shape (N_v1,); blob has shape (roi_size, roi_size)
        maps += v1_weights[:, j, None, None] * blob[None, :, :]

    return maps


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_kernel_grid(
    maps: np.ndarray,
    out_dir: Path,
    filename: str,
    suptitle: str,
    label_prefix: str,
    max_neurons: int = 64,
    ncols: int = 8,
    centroids: np.ndarray | None = None,
) -> None:
    """Save a grid of 2-D maps with a shared diverging colormap.

    If centroids is provided (shape (N, 2) as (row, col)), a crosshair is
    drawn at each neuron's centroid position.
    """
    N = maps.shape[0]
    n_show = min(N, max_neurons)
    nrows = (n_show + ncols - 1) // ncols

    vmax = np.abs(maps[:n_show]).max()
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.array(axes).reshape(nrows, ncols)

    crosshair_kw = dict(color="k", linewidth=0.6, alpha=0.7)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        ax.axis("off")
        if idx < n_show:
            ax.imshow(
                maps[idx],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            if centroids is not None:
                cr, cc = centroids[idx]
                ax.axhline(cr, **crosshair_kw)
                ax.axvline(cc, **crosshair_kw)
            ax.set_title(f"{label_prefix} {idx}", fontsize=7)

    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_v1_connection_maps(
    maps: np.ndarray,
    out_dir: Path,
    max_neurons: int = 64,
    ncols: int = 8,
) -> None:
    """Save a grid image of V1 connection maps."""
    _plot_kernel_grid(
        maps, out_dir,
        filename="v1_connection_maps.png",
        suptitle="V1 neuron RGC connection maps\n(red=positive, blue=negative weight)",
        label_prefix="V1",
        max_neurons=max_neurons,
        ncols=ncols,
    )


def plot_rgc_spatial_kernels(
    spatial_weights: np.ndarray,
    centroids: np.ndarray,
    out_dir: Path,
    max_neurons: int = 64,
    ncols: int = 8,
) -> None:
    """Save a grid image of raw RGC spatial kernels with centroid crosshairs."""
    _plot_kernel_grid(
        spatial_weights, out_dir,
        filename="rgc_spatial_kernels.png",
        suptitle="RGC spatial kernels\n(red=positive, blue=negative weight)",
        label_prefix="RGC",
        max_neurons=max_neurons,
        ncols=ncols,
        centroids=centroids,
    )


def plot_rgc_centroids(
    centroids: np.ndarray,
    roi_size: int,
    out_dir: Path,
) -> None:
    """Scatter plot of all RGC centroids."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(centroids[:, 1], centroids[:, 0], s=10, alpha=0.6, c="steelblue")
    ax.set_xlim(-0.5, roi_size - 0.5)
    ax.set_ylim(roi_size - 0.5, -0.5)   # image convention: row 0 at top
    ax.set_aspect("equal")
    ax.set_xlabel("Column (pixels)")
    ax.set_ylabel("Row (pixels)")
    ax.set_title(f"RGC spatial kernel centroids  (N={len(centroids)})")
    path = out_dir / "rgc_centroids.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint file")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian spread around each centroid (pixels, default 1.0)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: same folder as checkpoint)")
    parser.add_argument("--max-neurons", type=int, default=64, help="Max V1 neurons to plot in grid (default 64)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    out_dir = args.out or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    iteration = ckpt.get("iteration", "?")
    state = ckpt["model_state_dict"]
    print(f"  Iteration: {iteration}")

    # ------------------------------------------------------------------
    # Reconstruct weight tensors directly from the state dict
    # (avoids needing to know all constructor args)
    # ------------------------------------------------------------------
    # RGC spatial: (N_rgc, X*X)
    rgc_spatial_flat = state["rgc_encoder.spatial_projection.weight"]  # (N_rgc, X*X)
    N_rgc = rgc_spatial_flat.shape[0]
    X = int(round(rgc_spatial_flat.shape[1] ** 0.5))
    rgc_spatial = rgc_spatial_flat.view(N_rgc, X, X)

    # V1 spatial: (N_v1, N_rgc)
    v1_weights = state["v1_decoder.spatial_projection.weight"]  # (N_v1, N_rgc)
    N_v1 = v1_weights.shape[0]

    print(f"  RGC channels : {N_rgc},  roi_size : {X}")
    print(f"  V1  channels : {N_v1}")

    # ------------------------------------------------------------------
    # Compute centroids
    # ------------------------------------------------------------------
    print("Computing RGC centroids...")
    centroids = compute_rgc_centroids(rgc_spatial)

    # ------------------------------------------------------------------
    # Build V1 connection maps
    # ------------------------------------------------------------------
    print(f"Building V1 connection maps (sigma={args.sigma} px)...")
    v1_w_np = v1_weights.detach().cpu().float().numpy()  # (N_v1, N_rgc)
    maps = build_v1_connection_maps(centroids, v1_w_np, roi_size=X, sigma=args.sigma)

    # Save raw arrays for further analysis
    np.save(out_dir / "v1_connection_maps.npy", maps)
    print(f"  Saved raw maps: {out_dir / 'v1_connection_maps.npy'}  shape={maps.shape}")

    np.save(out_dir / "rgc_centroids.npy", centroids)
    print(f"  Saved centroids: {out_dir / 'rgc_centroids.npy'}  shape={centroids.shape}")

    rgc_spatial_np = rgc_spatial.float().numpy()  # (N_rgc, X, X)
    np.save(out_dir / "rgc_spatial_kernels.npy", rgc_spatial_np)
    print(f"  Saved RGC spatial kernels: {out_dir / 'rgc_spatial_kernels.npy'}  shape={rgc_spatial_np.shape}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("Generating plots...")
    plot_rgc_centroids(centroids, roi_size=X, out_dir=out_dir)
    plot_rgc_spatial_kernels(rgc_spatial_np, centroids, out_dir=out_dir, max_neurons=args.max_neurons)
    plot_v1_connection_maps(maps, out_dir=out_dir, max_neurons=args.max_neurons)

    print("\nDone.")


if __name__ == "__main__":
    main()
