import numpy as np
from typing import Tuple


def generate_brownian_motion(D: float, fs: int, length: int) -> np.ndarray:
    """
    Generates a 2D Brownian motion trace.

    Args:
        D (float): Diffusion coefficient.
        fs (int): Sampling frequency.
        length (int): Length of the trace (number of samples).

    Returns:
        np.ndarray: A 2D array of shape (2, length) representing the x and y
                    coordinates of the Brownian motion over time.
    """
    K = np.sqrt(2.0 * D / fs)
    eye_trace = K * np.random.randn(2, length - 1)
    eye_trace = np.concatenate([np.zeros((2, 1)), eye_trace], axis=1)
    eye_trace = np.cumsum(eye_trace, axis=1)
    return eye_trace


def generate_saccade(
    amplitude_deg: float, direction_deg: float, sampling_frequency_hz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D saccade trajectory based on amplitude, direction, and sampling frequency.

    Parameters:
    - amplitude_deg: float, amplitude of saccade in degrees
    - direction_deg: float, direction of saccade in degrees (0 = right, 90 = up)
    - sampling_frequency_hz: float, number of samples per second

    Returns:
    - time: np.ndarray, time vector in seconds
    - x: np.ndarray, horizontal position over time in degrees
    - y: np.ndarray, vertical position over time in degrees
    - velocity_profile: np.ndarray, velocity over time in deg/s
    """
    direction_rad = np.deg2rad(direction_deg)

    # Saccade main sequence estimates
    peak_velocity = 20 * np.sqrt(amplitude_deg) + 50  # deg/s
    duration_ms = 2.2 * amplitude_deg + 21  # ms
    duration_sec = duration_ms / 1000

    samples = int(duration_sec * sampling_frequency_hz)
    time = np.linspace(0, duration_sec, samples)

    # Normalized time (0 to 1)
    t_norm = time / duration_sec

    # Generate velocity profile (bell-shaped using beta-like function)
    beta_profile = 30 * t_norm**2 * (1 - t_norm) ** 2
    beta_profile /= np.max(beta_profile)

    velocity_profile = peak_velocity * beta_profile
    position_profile = np.cumsum(velocity_profile) / sampling_frequency_hz

    # Project to 2D
    x = position_profile * np.cos(direction_rad)
    y = position_profile * np.sin(direction_rad)

    return time, x, y, velocity_profile
