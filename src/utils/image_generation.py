import numpy as np


def pink_noise_gray_image(
    size: int, alpha: float = 1.0, return_white: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
        return_white (bool): If True, also return the uncolored (white noise)
                    image generated from the same random phases.

    Returns:
        np.ndarray: Pink noise image of shape (size, size).
        If return_white=True, returns (pink_img, white_img) where white_img is
        the spatial-domain image before 1/f spectrum coloring.
    """
    k = np.fft.fftfreq(size)
    k[0] = 1.0
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    H = 1.0 / (kr**alpha)
    H[0, 0] = 0.0
    phases = np.exp(1j * 2 * np.pi * np.random.randn(size, size))

    pink_img = np.real(np.fft.ifft2(H * phases))

    if return_white:
        white_img = np.real(np.fft.ifft2(phases))
        return pink_img, white_img

    return pink_img


def spatiotemporal_pink_noise(
    n_frames: int, size: int, alpha: float = 1.0
) -> np.ndarray:
    """
    Generate a spatiotemporal pink-noise volume.

    Produces a (n_frames, size, size) volume whose 3D power spectrum follows a
    1/f^alpha law over the combined temporal and spatial frequency magnitude.
    This is the structureless control counterpart to eye-movement-modulated
    input: there is no fixation/saccade trajectory, only isotropic 1/f noise in
    space and time.

    Args:
        n_frames (int): Number of temporal frames (T).
        size (int): Spatial side length of each square frame.
        alpha (float): Power-law exponent. alpha=1 is pink, alpha=0 is white.

    Returns:
        np.ndarray: Volume of shape (n_frames, size, size), float32, scaled to
        unit standard deviation.
    """
    ft = np.fft.fftfreq(n_frames)
    fs = np.fft.fftfreq(size)

    # Radial frequency magnitude over (t, y, x)
    fr = np.sqrt(
        ft[:, None, None] ** 2
        + fs[None, :, None] ** 2
        + fs[None, None, :] ** 2
    )
    fr[0, 0, 0] = 1.0  # avoid divide-by-zero at DC

    H = 1.0 / (fr**alpha)
    H[0, 0, 0] = 0.0  # remove DC component

    phases = np.exp(1j * 2 * np.pi * np.random.rand(n_frames, size, size))
    volume = np.real(np.fft.ifftn(H * phases))

    volume = volume / (volume.std() + 1e-8)
    return volume.astype(np.float32)
