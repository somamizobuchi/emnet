import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.stats import norm
import torch

matplotlib.use("agg")


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def kernel_images(W, kernel_size, image_channels, rows=None, cols=None, spacing=1):
    """
    Return the kernels as tiled images for visualization
    :return: np.ndarray, shape = [rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing, 1]
    """

    W /= np.linalg.norm(W, axis=0, keepdims=True)
    W = W.reshape(image_channels, -1, W.shape[-1])

    if rows is None:
        rows = int(np.ceil(math.sqrt(W.shape[-1])))
    if cols is None:
        cols = int(np.ceil(W.shape[-1] / rows))

    kernels = np.ones([3, rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing], dtype=np.float32)
    coords = [(i, j) for i in range(rows) for j in range(cols)]

    Wt = W.transpose(2, 0, 1)

    for (i, j), weight in zip(coords, Wt):
        kernel = weight.reshape(image_channels, kernel_size, kernel_size) * 2 + 0.5
        x = i * (kernel_size + spacing)
        y = j * (kernel_size + spacing)
        kernels[:, x:x+kernel_size, y:y+kernel_size] = kernel

    return kernels.clip(0, 1)


def plot_convolution(weight: torch.Tensor):
    if torch.is_tensor(weight):
        weight = weight.numpy()
    weight = weight / np.linalg.norm(weight, axis=-1, keepdims=True)

    fig = plt.figure(figsize=(4, 4))
    plt.plot(weight[:, 0, :].T)
    plt.tight_layout()
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ncol, nrow = fig.canvas.get_width_height()
    buf = buf.reshape(ncol, nrow, 3)
    plt.close()

    return buf.transpose(2, 0, 1)


## create natural noise image
def natural_noise(size):
    im = np.random.normal(0.0, 100.0, (size, size))

    kx = np.arange(-size/2, size/2)
    ky = kx.reshape(-1, 1)

    kx[kx == 0] = 1
    ky[ky == 0] = 1
    kr = 1. / np.sqrt(kx**2 + ky**2)

    Im = np.fft.fftshift(np.fft.fft2(im))
    Im = kr * Im;

    im = np.real(np.fft.ifft2(np.fft.fftshift(Im)))

    return im


def brownian_eye_trace(D: np.double, fs: int, n: int, seed: int = None) -> np.array:
    """
    Creates simulated eye traces based on brownian motion

    Parameters
    ----------
    D : double
        diffusion constant in arcmin^2/sec
    fs : int 
        sampling frequency in Hz
    n : int
        number of samples to generate
    
    Returns
    -------
    tuple
        A 2-by-n array of eye traces for x and y eye traces
    """
    if seed:
        rng = np.random.default_rng(seed=seed)
        trace = rng.normal(0., 1., (2, n))
    else:
        trace = np.random.normal(0., 1., (2,n))
    K = np.sqrt(2.*D / fs)
    return np.cumsum(K * trace, axis=1)

    
def crop_image(img, roi_size, center):
    return img[center[1]-roi_size//2:center[1]+roi_size//2, center[0]-roi_size//2:center[0]+roi_size//2]

def implay(seq, interval = 20, repeat = False, repeat_delay = -1):
    """
    Plays a sequence of gray images (2D arrays)

    Parameters
    ----------
    seq : Array
        The input sequence (x, y, t)
    interval : int
        Interval between frames in milliseconds
    """
    fig, ax = plt.subplots()
    video = []
    for i in range(0, seq.shape[2]):
        roi = seq[:,:,i]
        implt = ax.imshow(roi, animated=True, cmap='gray');
        if i == 0:
            ax.imshow(roi, cmap='gray')
        video.append([implt])

    ani = animation.ArtistAnimation(fig, video, interval=interval, blit=True, repeat=repeat, repeat_delay=repeat_delay)
    plt.show()

    
def generate_saccade(amplitude_deg: float, angle_radians: float, fs: int = 1000):
    """
    """
    # From gaussian
    peak_velocity = 150 * np.sqrt(amplitude_deg)
    sigma = 1 / ((peak_velocity / amplitude_deg) * np.sqrt(2*np.pi))
    t = np.arange(-sigma*3, sigma*3, 1/fs)
    pos = amplitude_deg * norm.cdf(t, loc=0, scale=sigma)  
    x = np.cos(angle_radians) * pos
    y = np.sin(angle_radians) * pos
    return np.vstack((x, y))

    
    
    
    