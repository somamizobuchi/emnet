import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
from tqdm import tqdm
from utils.eye_movement import generate_brownian_motion, generate_saccade
from utils.image_generation import pink_noise_gray_image


class PairedReconDataset(Dataset):
    """
    Dataset that returns paired pink noise and white noise frames along the
    same eye trace. Pink and white images are generated from the same random
    phases so they share the same spatial structure — only the spectrum differs.

    Each sample returns frames extracted at identical eye positions from both
    the pink-noise image and its uncolored (white noise) counterpart.
    """

    def __init__(
        self,
        img_size: int = 256,
        roi_size: int = 32,
        total_samples: int = 128,
        pad_start: int = 32,
        diffusion_coefficient: float = 20 / 3600,
        sampling_frequency: int = 360,
        pixels_per_degree: int = 240,
        saccade: bool = False,
    ):
        self.img_size = img_size
        self.roi_size = roi_size
        self.total_samples = total_samples
        self.pad_start = pad_start
        self.diffusion_coefficient = diffusion_coefficient
        self.sampling_frequency = sampling_frequency
        self.pixels_per_degree = pixels_per_degree
        self.saccade = saccade

        pink_imgs, white_imgs = [], []
        for _ in tqdm(range(500), desc="Generating images"):
            pink, white = pink_noise_gray_image(img_size, return_white=True)
            # Normalise each image pair to unit std (using pink std so the
            # same scale factor is applied to both)
            scale = pink.std() + 1e-8
            pink_imgs.append((pink / scale).astype(np.float32))
            white_imgs.append((white / scale).astype(np.float32))

        self.pink_imgs = np.stack(pink_imgs, axis=0)
        self.white_imgs = np.stack(white_imgs, axis=0)

    def __len__(self) -> int:
        return 1_000_000

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int]:
        """
        Generate a sample of paired pink/white frames along an eye trace.

        Returns:
            - pink_frames  (total_samples, roi_size, roi_size): pink noise patches
            - white_frames (total_samples, roi_size, roi_size): white noise patches
                           at the same eye positions
            - pink_img     (img_size, img_size): full pink noise image
            - white_img    (img_size, img_size): full white noise image
            - eye_trace    (2, total_samples): (x, y) eye positions
            - mask         (img_size, img_size): visited-region mask
            - sacc_end_idx (int): frame index where saccade ends (0 if no saccade)
        """
        idx = np.random.randint(0, self.pink_imgs.shape[0])
        pink_img = self.pink_imgs[idx]
        white_img = self.white_imgs[idx]

        eye_trace, sacc_end_idx = self.generate_eye_trace()

        pink_frames = np.zeros(
            (self.total_samples, self.roi_size, self.roi_size), dtype=np.float32
        )
        white_frames = np.zeros_like(pink_frames)
        mask = np.full((self.img_size, self.img_size), False)

        for i in range(self.total_samples):
            x, y = eye_trace[:, i]
            pink_frames[i] = pink_img[y : y + self.roi_size, x : x + self.roi_size]
            white_frames[i] = white_img[y : y + self.roi_size, x : x + self.roi_size]
            if i >= sacc_end_idx and i >= self.pad_start:
                mask[y : y + self.roi_size, x : x + self.roi_size] = True

        return (
            torch.from_numpy(pink_frames),
            torch.from_numpy(white_frames),
            torch.from_numpy(pink_img),
            torch.from_numpy(white_img),
            eye_trace,
            mask,
            sacc_end_idx,
        )

    def generate_eye_trace(self) -> Tuple[np.ndarray, int]:
        start_point = np.random.randint(
            0, self.img_size - self.roi_size, size=(2, 1)
        )

        if not self.saccade:
            while True:
                d = generate_brownian_motion(
                    self.diffusion_coefficient,
                    self.sampling_frequency,
                    self.total_samples,
                )
                d = np.round(d * self.pixels_per_degree).astype(int) + start_point
                if np.all(d >= 0) and np.all(d < self.img_size - self.roi_size):
                    break
            return d, 0

        while True:
            d1 = generate_brownian_motion(
                self.diffusion_coefficient, self.sampling_frequency, self.pad_start
            )
            d1 = np.round(d1 * self.pixels_per_degree).astype(int) + start_point
            if np.all(d1 >= 0) and np.all(d1 < self.img_size - self.roi_size):
                break

        while True:
            end_point = np.random.randint(
                0, self.img_size - self.roi_size, size=(2, 1)
            )
            amp_val: float = float(
                np.linalg.norm(end_point - d1[:, -1]) / self.pixels_per_degree
            )
            theta_val: float = float(
                np.rad2deg(np.atan2(end_point[1] - d1[1, -1], end_point[0] - d1[0, -1]))
            )
            _, sx, sy, _ = generate_saccade(amp_val, theta_val, self.sampling_frequency)
            s = np.vstack((sx, sy))
            s = np.round(s * self.pixels_per_degree).astype(int) + d1[:, -1:]
            if np.all(s >= 0) and np.all(s < self.img_size - self.roi_size):
                break

        sacc_end_idx = d1.shape[1] + s.shape[1] - 1

        while True:
            d2 = generate_brownian_motion(
                self.diffusion_coefficient,
                self.sampling_frequency,
                self.total_samples - self.pad_start - s.shape[1],
            )
            d2 = np.round(d2 * self.pixels_per_degree).astype(int) + s[:, -1:]
            if np.all(d2 >= 0) and np.all(d2 < self.img_size - self.roi_size):
                break

        return np.concatenate((d1, s, d2), axis=1), sacc_end_idx
