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
    data_dir = Path(config.root_dir) / 'fmow_v1.0'
    image_dir = Path(config.root_dir) / 'fmow_v1.0_images_jpg'
    os.makedirs(image_dir, exist_ok=True)

    img_counter = 0
    for chunk in tqdm(range(101)):
        npy_chunk = np.load(data_dir / f'rgb_all_imgs_{chunk}.npy', mmap_mode='r')
        for i in range(len(npy_chunk)):
            npy_image = npy_chunk[i]
            img = Image.fromarray(npy_image, mode='RGB')
            img.save(image_dir / f'rgb_img_{img_counter}.jpg')
            img_counter += 1

if __name__=='__main__':
    main()
