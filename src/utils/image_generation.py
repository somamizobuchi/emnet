import numpy as np


def pink_noise_gray_image(size: int, alpha: float = 1.0) -> np.ndarray:
    """
    Generates a pink noise (1/f) grayscale image of a given size.

    This function creates a pink noise image by generating white noise in the
    frequency domain and then applying a 1/f^alpha filter. The result is then
    transformed back to the spatial domain to produce the final image.

    Args:
        size (int): The size of the square image (width and height).
        alpha (float): The power-law exponent for the frequency filter.
                    alpha=1 corresponds to pink noise, alpha=0 to white noise,
                    and alpha=2 to Brownian noise.

    Returns:
        np.ndarray: A 2D numpy array representing the pink noise grayscale image.
    """
    k = np.fft.fftfreq(size)
    k[0] = 1.0
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    H = 1.0 / (kr**alpha)
    H[0, 0] = 0.0
    Im = H * np.exp(1j * 2 * np.pi * np.random.randn(size, size))

    return np.real(np.fft.ifft2(Im))
