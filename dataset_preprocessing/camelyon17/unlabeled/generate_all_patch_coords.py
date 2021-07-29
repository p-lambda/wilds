# Code adapted from https://github.com/liucong3/camelyon17
# and https://github.com/cv-lee/Camelyon17

import argparse
import os
import pdb

import openslide
import cv2
import numpy as np
import pandas as pd

CENTER_SIZE = 32
MASK_LEVEL = 4
PATCH_LEVEL = 2

NUM_OF_HOSPITALS = 5


def _make_masks(slide_path, mask_level, **args):
    """
    Return a slide with annotated tumor, normal, and tissue masks using an Otsu threshold
    """
    print("_make_masks(%s)" % slide_path)

    # Load slide
    slide = openslide.OpenSlide(slide_path)
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[mask_level]))

    # draw tissue mask
    slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
    slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
    slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
    slide_lv = slide_lv[:, :, 1]
    _, tissue_mask = cv2.threshold(
        slide_lv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return slide, slide_map, tissue_mask


def _write_masks(mask_folder_path, slide_map, tissue_mask, **args):
    """
    Write masks out to disk; used for sanity checking and visualization.
    """
    print("_write_masks")
    os.makedirs(mask_folder_path, exist_ok=True)
    map_path = os.path.join(mask_folder_path, "map.png")
    cv2.imwrite(map_path, slide_map)
    tissue_mask_path = os.path.join(mask_folder_path, "tissue_mask.png")
    cv2.imwrite(tissue_mask_path, np.array(tissue_mask))


def _record_patches(
    center_size,
    slide,
    slide_map,
    patch_level,
    mask_level,
    tissue_mask,
    normal_threshold,
    **args,
):
    """
    Extract all patches using the tissue masks.
    """
    width, height = np.array(slide.level_dimensions[patch_level]) // center_size
    print("_record_patches(w=%d,h=%d)" % (width, height))
    margin = 5
    mask_max = 255
    assert mask_level >= patch_level
    width_mask_step = (
        center_size
        * slide.level_dimensions[mask_level][0]
        / slide.level_dimensions[patch_level][0]
    )
    height_mask_step = (
        center_size
        * slide.level_dimensions[mask_level][1]
        / slide.level_dimensions[patch_level][1]
    )

    patch_list = []

    # These mark the coordinates of the central region of the patch
    for i in range(margin, width - margin):
        for j in range(margin, height - margin):
            # We no longer have access to the tumor and normal masks. Just use the tissue mask
            mask_i_start = round(width_mask_step * i)
            mask_i_end = round(width_mask_step * (i + 1))
            mask_j_start = round(height_mask_step * j)
            mask_j_end = round(height_mask_step * (j + 1))

            # Compute mask only over central region
            tissue_mask_avg = tissue_mask[
                mask_j_start:mask_j_end, mask_i_start:mask_i_end
            ].mean()
            tissue_area_ratio = tissue_mask_avg / mask_max

            # Tissue is the union of normal and tumor, so check the tissue area ratio is above the normal threshold
            if tissue_area_ratio > normal_threshold:
                # Set the label to be -1 to indicate it's unlabeled data
                patch_list.append((center_size * i, center_size * j, -1))
                cv2.rectangle(
                    slide_map,
                    (mask_i_start, mask_j_start),
                    (mask_i_end, mask_j_end),
                    (100, 149, 237),  # cornflower blue for debugging
                    thickness=1,
                )

    print(f"Added {len(patch_list)} patches...")
    df = pd.DataFrame(patch_list, columns=["x_coord", "y_coord", "tumor"])
    return df


def generate_file(patient, node, slide_path, folder_path):
    args = {
        "slide_path": slide_path,
        "patch_level": PATCH_LEVEL,
        "mask_level": MASK_LEVEL,
        "center_size": CENTER_SIZE,
        "mask_folder_path": folder_path,
        "normal_threshold": 0.2,
    }
    args["slide"], args["slide_map"], args["tissue_mask"] = _make_masks(**args)
    df = _record_patches(**args)
    df["patient"] = patient
    df["node"] = node
    _write_masks(**args)
    return df


def generate_files(slide_root, output_root, center):
    aggregate_df = pd.DataFrame(
        columns=["patient", "node", "x_coord", "y_coord", "tumor"]
    )

    # Assume files are organized in the following way:
    #   center_<center#>/patient_<patient#>/patient_<patient#>_node_<node#>.tif
    if center is None:
        print(
            "A value for --center was not specified. Generating patches for all centers..."
        )
        centers = range(NUM_OF_HOSPITALS)
    else:
        centers = [center]

    for center in centers:
        center_dir = os.path.join(slide_root, f"center_{center}")
        patient_dirs = os.listdir(center_dir)

        for patient_dir in patient_dirs:
            patient_dir = os.path.join(center_dir, patient_dir)
            if not os.path.isdir(patient_dir):
                continue

            for slide_file in os.listdir(patient_dir):
                if not slide_file.endswith(".tif"):
                    continue

                slide_path = os.path.join(patient_dir, slide_file)
                prefix = slide_file.split(".tif")[0]
                try:
                    assert len(prefix.split("_")) == 4

                    # The XML files have labels so it's not needed for Unlabeled Camelyon
                    df = generate_file(
                        patient=prefix.split("_")[1],
                        node=prefix.split("_")[3],
                        slide_path=slide_path,
                        folder_path=os.path.join(output_root, "masks", prefix),
                    )
                    aggregate_df = pd.concat([aggregate_df, df])
                except openslide.OpenSlideError as err:
                    print(err)
                    continue

    # Coordinates of all potential patches
    aggregate_df = aggregate_df.reset_index(drop=True)
    aggregate_df.to_csv(os.path.join(output_root, "all_patch_coords.csv"))
    return aggregate_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--slide_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--center",
        type=int,
        help="Which specific center to extract patches for. If a center is not specified, "
             "patches will be extracted for all five centers.",
    )
    args = parser.parse_args()

    generate_files(
        slide_root=args.slide_root, output_root=args.output_root, center=args.center
    )
