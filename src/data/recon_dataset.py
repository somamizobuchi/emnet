import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
from utils.eye_movement import generate_brownian_motion, generate_saccade
from utils.reconstruction import stitch_frames_by_position
from utils.image_generation import pink_noise_gray_image


class ReconDataset(Dataset):
    """
    A dataset class for loading and processing reconstruction data.
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
        average: bool = False,
        use_pink: bool = True,
    ):
        """Initialize the dataset with parameters."""
        self.img_size = img_size
        self.roi_size = roi_size
        self.total_samples = total_samples
        self.pad_start = pad_start
        self.diffusion_coefficient = diffusion_coefficient
        self.sampling_frequency = sampling_frequency
        self.pixels_per_degree = pixels_per_degree
        self.saccade = saccade
        self.average = average
        self.use_pink = use_pink

        pink_imgs = []
        white_imgs = []
        print("Generating images...")
        for _ in range(500):
            pink, white = pink_noise_gray_image(img_size, return_white=True)
            pink = pink / (pink.std() + 1e-8)
            white = white / (white.std() + 1e-8)
            pink_imgs.append(pink)
            white_imgs.append(white)
        self.imgs = np.stack(pink_imgs, axis=0).astype(np.float32)
        self.white_imgs = np.stack(white_imgs, axis=0).astype(np.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return 1_000_000

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int]:
        """
        Generate a sample consisting of video frames captured along an eye trace.

        This method simulates eye movement over a pink noise image by generating
        an eye trace (fixation, saccade, or both) and extracting ROI patches at
        each timestep along the trajectory.

        Args:
            index: Sample index (unused, data is randomly generated each time).

        Returns:
            A tuple containing:
            - video_frames (torch.Tensor): Sequence of ROI patches of shape
              (total_samples, roi_size, roi_size) extracted along the eye trace.
            - img (torch.Tensor): Target image of shape (img_size, img_size).
              If average=True, this is the averaged ROI patches; otherwise it's
              the original pink noise image.
            - eye_trace (np.ndarray): 2D array of shape (2, total_samples)
              containing (x, y) coordinates of the eye position over time.
            - mask (np.ndarray): Boolean array of shape (img_size, img_size)
              indicating which regions were visited after saccade completion
              and padding.
            - sacc_end_idx (int): Index where the saccade ends (0 if no saccade).
        """
        idx = np.random.randint(0, self.imgs.shape[0])
        pink_img = self.imgs[idx]
        img = pink_img if self.use_pink else self.white_imgs[idx]

        # Generate eye trace
        eye_trace, sacc_end_idx = self.generate_eye_trace()

        # Generate video frames and target image based on the eye trace
        video_frames = np.zeros(
            (self.total_samples, self.roi_size, self.roi_size), dtype=np.float32
        )

        if self.average:
            w = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            w = None

        mask = np.full((self.img_size, self.img_size), False)
        for i in range(self.total_samples):
            x, y = eye_trace[:, i]
            video_frames[i] = pink_img[y : y + self.roi_size, x : x + self.roi_size]
            if i >= sacc_end_idx and i >= self.pad_start:
                if self.average and w is not None:
                    w[y : y + self.roi_size, x : x + self.roi_size] += video_frames[i]
                mask[y : y + self.roi_size, x : x + self.roi_size] = True

        if self.average and w is not None:
            pink_img = w / (self.total_samples - self.pad_start)
            img = pink_img if self.use_pink else img

        return (
            torch.from_numpy(video_frames),
            torch.from_numpy(img),
            eye_trace,
            mask,
            sacc_end_idx,
        )

    def generate_eye_trace(self) -> Tuple[np.ndarray, int]:
        start_point = np.random.randint(
            0, self.img_size - self.roi_size, size=(2, 1)
        )  # Random starting point

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
            )  # Random end point
            # Generate saccades
            amp_val: float = float(
                np.linalg.norm(end_point - d1[:, -1]) / self.pixels_per_degree
            )
            theta_val: float = float(
                np.rad2deg(np.atan2(end_point[1] - d1[1, -1], end_point[0] - d1[0, -1]))
            )  # Angle in degrees
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

        # Combine drift and saccade
        return (np.concatenate((d1, s, d2), axis=1), sacc_end_idx)
