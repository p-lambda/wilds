import os, sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
    config = parser.parse_args()
    data_dir = Path(config.root_dir) / 'poverty_v1.0'
    indiv_dir = Path(config.root_dir) / 'poverty_v1.0_indiv_npz'
    os.makedirs(indiv_dir, exist_ok=True)

    f = np.load(data_dir / 'landsat_poverty_imgs.npy', mmap_mode='r')
    f = f.transpose((0, 3, 1, 2))
    for i in tqdm(range(len(f))):
        x = f[i]
        np.savez_compressed(indiv_dir / f'landsat_poverty_img_{i}.npz', x=x)

if __name__=='__main__':
    main()
