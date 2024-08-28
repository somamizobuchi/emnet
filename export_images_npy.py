import os
from pathlib import Path
import numpy as np
from scipy.io.matlab import loadmat
from tqdm import tqdm
from glob import glob

def main():
    root = "upenn"
    files = sorted(glob(f"{root}/*.mat"))

    for file in tqdm(files, "Saving to npy"):
        im = loadmat(file)['LUM_Image'].astype(np.float32)
        filepath = Path(file);
        im -= im.mean()
        im /= im.std()
        np.save(os.path.join(filepath.parent, filepath.stem + ".npy"), im)

if __name__ == "__main__":
    main()