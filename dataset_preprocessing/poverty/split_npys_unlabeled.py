import os, sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True,
                        help='The poverty data directory.')
    parser.add_argument('--out_dir_root', required=True,
                        help='The directory where output dir should be made.')
    args = parser.parse_args()

    data_dir = Path(args.root_dir)
    indiv_dir = Path(args.out_dir_root) / 'poverty_unlabeled_v1.0_indiv_npz' / 'images'
    indiv_dir.mkdir(exist_ok=True, parents=True)

    counter = 0
    for i in range(27):
        path = data_dir / f'unlabeled_landsat_poverty_imgs_{i}.npy'
        arr = np.load(path, mmap_mode='r')
        arr = arr.transpose((0, 3, 1, 2))
        for j in tqdm(range(len(arr))):
            x = arr[j]
            np.savez_compressed(indiv_dir / f'landsat_poverty_img_{counter}.npz', x=x)
            counter += 1


if __name__=='__main__':
    main()
