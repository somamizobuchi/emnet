import os
from tqdm import tqdm
from scipy.io.matlab import loadmat
from glob import glob
from util import crop_image, brownian_eye_trace, implay
import numpy as np
import matplotlib
import uuid


matplotlib.use('Qt5Agg')


def main():
    files = glob("upenn_fixations/*")
    for f in tqdm(files, "Removing files"):
        os.remove(f)


    fixations_per_image = 3;
    root = "upenn"
    roi_size = 512
    samples = 100
    fs = 300    # sampling frequency
    ppd = 180   # pixels per degree
    diffusion_constant = 20 # arcmin^2/sec
    pad_pixels = 30 # samples to leave around edges in case fixations exceed

    # Load mat file
    files = [mat for mat in os.listdir(root) if mat.endswith('.mat')]
    print("Loading {} images from {} ...".format(len(files), root))
    for file in tqdm(files):
        image = loadmat(os.path.join(root, file))['LUM_Image'].astype(np.float32)
        std = np.std(image)
        if std < 1e-4:
            continue
        image -= np.mean(image)
        image /= std

        max_idx =  np.array(image.shape) - roi_size - pad_pixels
        
        # Generate fixations
        video = np.zeros((samples,roi_size, roi_size), np.float32)
        for fixation in range(fixations_per_image):
            trace = brownian_eye_trace(diffusion_constant, fs, samples) / 60
            offset  = np.array((np.random.randint(pad_pixels, max_idx[0]),
                        np.random.randint(pad_pixels, max_idx[1])))
            trace = np.round(trace * ppd + offset[:, np.newaxis]).astype(int)
            for frame in range(trace.shape[1]):
                x_start = trace[1, frame]
                y_start = trace[0, frame]
                video[frame,:,:] = image[y_start:(y_start+roi_size), x_start:(x_start+roi_size)]
            
            np.save("upenn_fixations/{}.npy".format(uuid.uuid4()), video)
            
            
            
        

if __name__ == "__main__":
    main()