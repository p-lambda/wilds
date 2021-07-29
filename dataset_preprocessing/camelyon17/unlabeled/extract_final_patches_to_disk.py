import argparse
import os
import pdb
from tqdm import tqdm

import openslide
import pandas as pd

from generate_all_patch_coords import PATCH_LEVEL, CENTER_SIZE


def write_patch_images_from_df(slide_root, output_root):
    read_df = pd.read_csv(
        os.path.join(output_root, "metadata.csv"), index_col=0, dtype={"patient": "str"}
    )

    patch_level = PATCH_LEVEL
    center_size = CENTER_SIZE
    patch_size = center_size * 3

    for idx in tqdm(read_df.index):
        orig_x = read_df.loc[idx, "x_coord"]
        orig_y = read_df.loc[idx, "y_coord"]
        center = read_df.loc[idx, "center"]
        patient = read_df.loc[idx, "patient"]
        node = read_df.loc[idx, "node"]

        patch_folder = os.path.join(
            output_root, "patches", f"patient_{patient}_node_{node}"
        )
        patch_path = os.path.join(
            patch_folder,
            f"patch_patient_{patient}_node_{node}_x_{orig_x}_y_{orig_y}.png",
        )

        os.makedirs(patch_folder, exist_ok=True)
        if os.path.isfile(patch_path):
            continue

        slide_path = os.path.join(
            slide_root,
            f"center_{center}",
            f"patient_{patient}",
            f"patient_{patient}_node_{node}.tif",
        )
        slide = openslide.OpenSlide(slide_path)

        # Coords are at patch_level
        # First shift coords to top left corner of the entire patch
        x = orig_x - center_size
        y = orig_y - center_size
        # Then match to level 0 coords so we can use read_region
        x = int(
            round(
                x
                * slide.level_dimensions[0][0]
                / slide.level_dimensions[patch_level][0]
            )
        )
        y = int(
            round(
                y
                * slide.level_dimensions[0][1]
                / slide.level_dimensions[patch_level][1]
            )
        )

        patch = slide.read_region((x, y), 2, (patch_size, patch_size))
        patch.save(patch_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    write_patch_images_from_df(slide_root=args.slide_root, output_root=args.output_root)
