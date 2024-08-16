from util import generate_saccade, implay
import matplotlib
import matplotlib.pyplot as plt
from data import EMSequenceDataset, VideoDataset
import numpy as np
import torch

matplotlib.use("Qt5Agg")
kernel_size = 128 
frames = 20 
circle_masking = False
group_size = None
random_flip = False

dataset = EMSequenceDataset(kernel_size, frames, group_size=None, ppd=60, fsamp=300)

# dataset = VideoDataset("palmer", kernel_size, frames, circle_masking, group_size, random_flip)
im = dataset[100]
implay(np.transpose(im, (1, 2, 0)), repeat=True, repeat_delay=10)
plt.hist(im.flatten())
plt.show()

data_cov = dataset.covariance()
plt.imshow(data_cov)
plt.show()
print(data_cov.shape)
H_X = torch.linalg.cholesky(data_cov).diag().log2().sum().item() + data_cov.shape[0] / 2.0 * np.log2(2 * np.pi * np.e)




