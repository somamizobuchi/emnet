"""Train a grid of SimpleReconNet configurations and compare them.

This is not a hyperparameter *search*: every point on the grid is trained and
all results are kept for side-by-side comparison. Edit BASE_CONFIG and GRID
below to define an experiment, then run:

    python grid_train.py [max_iterations]

Outputs (under runs/grid/<timestamp>/):
    - one subdirectory per config with tensorboard logs + checkpoint
    - results.csv                  : final metrics for every config
    - compare_losses.png           : grouped bar chart of final losses
    - compare_spatial.png          : spatial kernel montage, one panel per config
    - compare_spatial_freq.png     : mean log |FFT|² of spatial kernels, one panel per config
    - compare_temporal.png         : temporal kernel overlay, one panel per config
    - compare_temporal_freq.png    : temporal kernel magnitude spectra, one panel per config
"""

import sys
import csv
import json
import itertools
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.simple_model import SimpleReconNet
from src.data.recon_dataset import ReconDataset
from src.data.stpink_dataset import STPinkDataset
from src.training.simple_trainer import SimpleTrainer
from src.utils import visualization as viz

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------
# Base config holds every parameter. GRID overrides a subset; the cartesian
# product of GRID's value lists defines the runs. Any key valid here can be
# placed in GRID.
BASE_CONFIG = {
    # model / data geometry
    "img_size": 64,
    "roi_size": 20,
    "fix_length": 64,
    "rgc_channels": 128,
    "rgc_temporal_length": 16,
    "rgc_delay": 1,
    "noise_std": 1.0,
    # dataset
    "input_mode": "eye",  # "eye" or "stpink"
    "stpink_alpha": 2.0,
    "diffusion_coefficient": 10.0 / 3600.0,
    "sampling_frequency": 360,
    "pixels_per_degree": 240,
    # losses
    "w_recon": 1.0,
    "w_sparsity": 0,
    "w_smoothness": 3e-2,
    "w_dc": 0,
    "w_kernel_var": 1e-3,
    "recon_loss": "l1",
    # optim
    "batch_size": 16,
    "learning_rate": 2e-3,
}

GRID = {"input_mode": ["eye", "stpink"], "noise_std": [1e-2, 1e-1, 1]}

# Default training length per config; override with the CLI arg.
DEFAULT_MAX_ITERATIONS = 20_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def expand_grid(base: dict, grid: dict) -> list[dict]:
    """Cartesian product of grid overrides applied on top of base."""
    if not grid:
        return [dict(base)]
    keys = list(grid.keys())
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(base)
        cfg.update(dict(zip(keys, values)))
        configs.append(cfg)
    return configs


def config_label(grid_keys: list[str], cfg: dict) -> str:
    """Short label built only from the parameters that vary across the grid."""
    parts = []
    for k in grid_keys:
        v = cfg[k]
        v_str = f"{v:g}" if isinstance(v, float) else str(v)
        parts.append(f"{k}={v_str}")
    return ", ".join(parts) if parts else "single"


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_dataset(cfg: dict):
    pad_start = cfg["rgc_temporal_length"] - 1
    if cfg["input_mode"] == "stpink":
        return STPinkDataset(
            roi_size=cfg["roi_size"],
            total_samples=cfg["fix_length"],
            alpha=cfg["stpink_alpha"],
        )
    if cfg["input_mode"] == "eye":
        return ReconDataset(
            img_size=cfg["img_size"],
            roi_size=cfg["roi_size"],
            total_samples=cfg["fix_length"],
            pad_start=pad_start,
            diffusion_coefficient=cfg["diffusion_coefficient"],
            sampling_frequency=cfg["sampling_frequency"],
            pixels_per_degree=cfg["pixels_per_degree"],
        )
    raise ValueError(f"Unknown input_mode: {cfg['input_mode']!r}")


def build_model(cfg: dict) -> SimpleReconNet:
    return SimpleReconNet(
        img_size=cfg["img_size"],
        roi_size=cfg["roi_size"],
        rgc_channels=cfg["rgc_channels"],
        rgc_temporal_length=cfg["rgc_temporal_length"],
        rgc_delay=cfg["rgc_delay"],
        noise_std=cfg["noise_std"],
    )


METRIC_KEYS = [
    "total_loss",
    "recon_loss",
    "sparsity_loss",
    "smoothness_loss",
    "dc_loss",
    "kernel_var_loss",
]


def run_one(
    cfg: dict, label: str, run_dir: Path, max_iterations: int, device: str
) -> dict:
    """Train a single config; return the final metrics dict."""
    dataset = build_dataset(cfg)
    model = build_model(cfg)

    trainer = SimpleTrainer(
        model=model,
        dataset=dataset,
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        w_recon=cfg["w_recon"],
        w_sparsity=cfg["w_sparsity"],
        w_smoothness=cfg["w_smoothness"],
        w_dc=cfg["w_dc"],
        w_kernel_var=cfg["w_kernel_var"],
        recon_loss=cfg["recon_loss"],
        device=device,
        log_dir=str(run_dir),
        save_every=max_iterations,  # one checkpoint at the end
        checkpoint_dir=str(run_dir / "checkpoints"),
    )

    # Train while keeping the last metrics dict (trainer.train doesn't return it)
    last_metrics: dict = {}
    pbar = tqdm(range(max_iterations), desc=label, leave=True)
    for _ in pbar:
        last_metrics = trainer.train_step()
        pbar.set_postfix(
            {
                "total": f"{last_metrics['total_loss']:.4f}",
                "recon": f"{last_metrics['recon_loss']:.4f}",
            }
        )
    trainer.save_checkpoint()
    if trainer.writer:
        trainer.writer.flush()
        trainer.writer.close()

    # Snapshot kernels for the comparison figures
    spatial, temporal = model.get_rgc_weights()
    np.save(run_dir / "spatial_kernels.npy", spatial.detach().cpu().numpy())
    np.save(run_dir / "temporal_kernels.npy", temporal.detach().cpu().numpy())

    return {k: float(last_metrics.get(k, float("nan"))) for k in METRIC_KEYS}


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
def plot_loss_comparison(labels: list[str], results: list[dict], out_path: Path):
    metrics = METRIC_KEYS
    n_cfg = len(labels)
    x = np.arange(len(metrics))
    width = 0.8 / max(n_cfg, 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(metrics)), 5))
    for i, (label, res) in enumerate(zip(labels, results)):
        vals = [res[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width * (n_cfg - 1) / 2)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("final loss value")
    ax.set_yscale("log")
    ax.set_title("Final losses by config")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _panel_grid(n: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_spatial_comparison(
    labels: list[str], run_dirs: list[Path], roi_size: int, out_path: Path
):
    n = len(labels)
    rows, cols = _panel_grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < n:
            spatial = torch.from_numpy(np.load(run_dirs[idx] / "spatial_kernels.npy"))
            grid = viz.plot_spatial_kernels(
                spatial.view(spatial.shape[0], -1), roi_size
            )
            ax.imshow(grid.squeeze(0).numpy(), cmap="RdBu_r")
            ax.set_title(labels[idx], fontsize=9)
        ax.axis("off")
    fig.suptitle("RGC spatial kernels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_temporal_comparison(labels: list[str], run_dirs: list[Path], out_path: Path):
    n = len(labels)
    rows, cols = _panel_grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < n:
            temporal = np.load(run_dirs[idx] / "temporal_kernels.npy")  # (N, T)
            for k in range(temporal.shape[0]):
                ax.plot(temporal[k], color="steelblue", alpha=0.15, linewidth=0.8)
            ax.axhline(0.0, color="k", linewidth=0.5)
            ax.set_title(labels[idx], fontsize=9)
            ax.set_xlabel("tap")
        else:
            ax.axis("off")
    fig.suptitle("RGC temporal kernels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _radial_average(power2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = power2d.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    fy = (np.arange(H) - cy) / H
    fx = (np.arange(W) - cx) / W
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    max_r = min(cy / H, cx / W)
    n_bins = min(H, W) // 2
    bins = np.linspace(0, max_r, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_power = np.array(
        [
            (
                power2d[(r >= bins[i]) & (r < bins[i + 1])].mean()
                if ((r >= bins[i]) & (r < bins[i + 1])).any()
                else 0.0
            )
            for i in range(n_bins)
        ]
    )
    return centers, mean_power


def plot_spatial_freq_comparison(
    labels: list[str], run_dirs: list[Path], roi_size: int, out_path: Path
):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, run_dir in zip(labels, run_dirs):
        spatial = np.load(run_dir / "spatial_kernels.npy")  # (N, H, W) or (N, H*W)
        N = spatial.shape[0]
        w = spatial.reshape(N, roi_size, roi_size)
        power = np.abs(np.fft.fftshift(np.fft.fft2(w), axes=(-2, -1))) ** 2
        profiles = np.array([_radial_average(power[i])[1] for i in range(N)])
        freqs, _ = _radial_average(power[0])
        ax.plot(freqs, profiles.mean(axis=0), linewidth=1.5, label=label)
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("Mean power")
    ax.set_title("RGC spatial kernels — radially averaged power spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_temporal_freq_comparison(
    labels: list[str],
    run_dirs: list[Path],
    sampling_frequency: float,
    out_path: Path,
):
    n = len(labels)
    rows, cols = _panel_grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < n:
            temporal = np.load(run_dirs[idx] / "temporal_kernels.npy")  # (N, T)
            T = temporal.shape[1]
            freqs = np.fft.rfftfreq(T, d=1.0 / sampling_frequency)
            spectra = np.abs(np.fft.rfft(temporal, axis=1))
            for row in spectra:
                ax.plot(freqs, row, color="steelblue", alpha=0.15, linewidth=0.8)
            ax.plot(
                freqs, spectra.mean(axis=0), color="navy", linewidth=1.5, label="mean"
            )
            ax.set_title(labels[idx], fontsize=9)
            ax.set_xlabel("Hz")
            ax.set_ylabel("|FFT|")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")
    fig.suptitle("RGC temporal kernels — frequency domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    max_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_ITERATIONS

    grid_keys = list(GRID.keys())
    configs = expand_grid(BASE_CONFIG, GRID)
    labels = [config_label(grid_keys, c) for c in configs]

    device = select_device()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    grid_root = Path(f"runs/grid/{timestamp}")
    grid_root.mkdir(parents=True, exist_ok=True)

    print(f"Grid root : {grid_root}")
    print(f"Device    : {device}")
    print(f"Configs   : {len(configs)}  x  {max_iterations} iterations each")
    print(f"Varying   : {grid_keys}")
    print("=" * 80)

    run_dirs: list[Path] = []
    results: list[dict] = []

    for i, (cfg, label) in enumerate(zip(configs, labels)):
        run_dir = grid_root / f"run_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        print(f"\n[{i + 1}/{len(configs)}] {label}")
        metrics = run_one(cfg, label, run_dir, max_iterations, device)
        print("  final: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        run_dirs.append(run_dir)
        results.append(metrics)

    # ---- results.csv ----
    csv_path = grid_root / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "label"] + grid_keys + METRIC_KEYS)
        for i, (cfg, label, res) in enumerate(zip(configs, labels, results)):
            row = [f"run_{i:02d}", label]
            row += [cfg[k] for k in grid_keys]
            row += [res[m] for m in METRIC_KEYS]
            writer.writerow(row)

    # ---- comparison figures ----
    roi_size = BASE_CONFIG["roi_size"]
    sampling_frequency = BASE_CONFIG["sampling_frequency"]
    plot_loss_comparison(labels, results, grid_root / "compare_losses.png")
    plot_spatial_comparison(
        labels, run_dirs, roi_size, grid_root / "compare_spatial.png"
    )
    plot_temporal_comparison(labels, run_dirs, grid_root / "compare_temporal.png")
    plot_spatial_freq_comparison(
        labels, run_dirs, roi_size, grid_root / "compare_spatial_freq.png"
    )
    plot_temporal_freq_comparison(
        labels, run_dirs, sampling_frequency, grid_root / "compare_temporal_freq.png"
    )

    print("\n" + "=" * 80)
    print(f"Done. Results in {grid_root}")
    print(f"  {csv_path}")
    print(f"  {grid_root / 'compare_losses.png'}")
    print(f"  {grid_root / 'compare_spatial.png'}")
    print(f"  {grid_root / 'compare_spatial_freq.png'}")
    print(f"  {grid_root / 'compare_temporal.png'}")
    print(f"  {grid_root / 'compare_temporal_freq.png'}")


if __name__ == "__main__":
    main()
