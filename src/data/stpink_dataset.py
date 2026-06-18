import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
from tqdm import trange

from utils.image_generation import spatiotemporal_pink_noise


class STPinkDataset(Dataset):
    """
    Spatiotemporal pink-noise dataset.

    Structureless control for ReconDataset: instead of extracting ROI patches
    along an eye-movement trajectory, each sample is a fresh (T, roi_size,
    roi_size) volume of 3D pink noise. The model reconstructs its own input, so
    the trainer's reconstruction target (frames[:, pad_start:]) is well-defined
    without any eye trace.

    The return signature matches ReconDataset so trainers can consume either
    dataset interchangeably: (video_frames, target_img, eye_trace, mask,
    sacc_end_idx). The trailing fields are placeholders here.

    At init, a pool of oversized volumes (cache_size × cache_t × cache_size ×
    cache_size) is pre-generated. Each __getitem__ crops a random
    (total_samples, roi_size, roi_size) window, giving effectively unlimited
    diversity with no per-sample FFT cost.
    """

    def __init__(
        self,
        roi_size: int = 32,
        total_samples: int = 128,
        alpha: float = 1.0,
        cache_size: int = 100,
        spatial_scale: int = 4,
        temporal_scale: int = 4,
    ):
        self.roi_size = roi_size
        self.total_samples = total_samples
        self.alpha = alpha

        cache_spatial = roi_size * spatial_scale
        cache_temporal = total_samples * temporal_scale

        print("Generating spatiotemporal pink noise cache...")
        volumes = []
        for _ in trange(cache_size):
            v = spatiotemporal_pink_noise(cache_temporal, cache_spatial, alpha)
            volumes.append(v)
        # (cache_size, cache_temporal, cache_spatial, cache_spatial)
        self.volumes = np.stack(volumes, axis=0)
        self._cache_t = cache_temporal
        self._cache_s = cache_spatial

    def __len__(self) -> int:
        return 1_000_000

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int]:
        vol_idx = np.random.randint(0, self.volumes.shape[0])
        t0 = np.random.randint(0, self._cache_t - self.total_samples + 1)
        y0 = np.random.randint(0, self._cache_s - self.roi_size + 1)
        x0 = np.random.randint(0, self._cache_s - self.roi_size + 1)

        video_frames = self.volumes[
            vol_idx,
            t0 : t0 + self.total_samples,
            y0 : y0 + self.roi_size,
            x0 : x0 + self.roi_size,
        ]  # (T, roi_size, roi_size) — already float32, already a view

        target_img = video_frames[-1]
        eye_trace = np.zeros((2, self.total_samples), dtype=np.int64)
        mask = np.zeros((self.roi_size, self.roi_size), dtype=bool)
        sacc_end_idx = 0

        return (
            torch.from_numpy(video_frames.copy()),
            torch.from_numpy(target_img.copy()),
            eye_trace,
            mask,
            sacc_end_idx,
        )
