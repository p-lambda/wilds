import argparse
import os
import pdb

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Fix seed for reproducibility
np.random.seed(0)

_NUM_CENTERS = 5
_NUM_PATCHES_TO_SUBSAMPLE = 6667
_NUM_PATIENTS_PER_HOSPITAL = 20


def generate_final_metadata(slide_root, output_root):
    def print_stats(patches_df):
        print(f"\nStatistics:\nTotal # of patches: {patches_df.shape[0]}")
        for center in range(_NUM_CENTERS):
            print(
                f"Center {center}: {np.sum(patches_df['center'] == center):6d} patches"
            )
        print()

    patches_path = os.path.join(output_root, "all_unlabeled_patch_coords.csv")
    print(f"Importing patches from {patches_path}...")
    df = pd.read_csv(
        patches_path,
        index_col=0,
        dtype={"patient": "str", "tumor": "int"},
    )

    # Assign slide numbers to patients + nodes
    patient_node_list = list(
        set(df[["patient", "node"]].itertuples(index=False, name=None))
    )
    patient_node_list.sort()
    patient_node_to_slide_map = {}
    for idx, (patient, node) in enumerate(patient_node_list):
        patient_node_to_slide_map[(patient, node)] = idx

    for (patient, node), slide_idx in patient_node_to_slide_map.items():
        mask = (df["patient"] == patient) & (df["node"] == node)
        df.loc[mask, "slide"] = slide_idx
    df["slide"] = df["slide"].astype("int")

    # The raw data has the following assignments:
    # Center 0: patients 0 to 19
    # Center 1: patients 20 to 39
    # Center 2: patients 40 to 59
    # Center 3: patients 60 to 79
    # Center 4: patients 80 to 99
    df["center"] = df["patient"].astype("int") // _NUM_PATIENTS_PER_HOSPITAL
    print_stats(df)
    for center, slide in set(
        df[["center", "slide"]].itertuples(index=False, name=None)
    ):
        assert center == slide // 100, "Expected 100 slides per center."

    # Remove patches from the original metadata.csv before subsampling.
    # There are 50 XML files in the lesion_annotation folder, so 50 patient-node pairs were
    # already used in the original WILDS Camelyon dataset.
    print(
        "Removing patches from slides that were used in the original Camelyon-WILDS dataset..."
    )
    for file in os.listdir(os.path.join(slide_root, "lesion_annotations")):
        if file.endswith(".xml") and not file.startswith("._"):
            prefix = file.split(".xml")[0]
            patient = prefix.split("_")[1]
            node = prefix.split("_")[3]

            patient_mask = df["patient"] == patient
            node_mask = df["node"] == int(node)
            df = df[~(patient_mask & node_mask)]
    print_stats(df)

    # The labeled Camelyon-WILDS dataset has approximately 300,000 patches. We want about 10x unlabeled data,
    # which corresponds to ~3 million patches. Since each hospital of the original Camelyon17 training set
    # has a 100 slides, we subsample 6,667 patches from each slide, resulting in 600,030 patches total from each
    # hospital except for Center 0. Slide 38 of Center 0 only has 5,824 patches, so we instead subsample a total of
    # 599,187 patches for Center 0. Therefore, there is a total of 2,999,307 unlabeled patches across the hospitals.
    print(f"Subsampling {_NUM_PATCHES_TO_SUBSAMPLE} patches from each slide...")
    indices_to_keep = []
    for slide in set(df["slide"]):
        slide_mask = df["slide"] == slide
        slide_indices = list(df.index[slide_mask])
        print(
            f"slide={slide}, choosing {_NUM_PATCHES_TO_SUBSAMPLE} patches from {len(slide_indices)} patches"
        )
        if _NUM_PATCHES_TO_SUBSAMPLE < len(slide_indices):
            indices_to_keep += list(
                np.random.choice(
                    slide_indices, size=_NUM_PATCHES_TO_SUBSAMPLE, replace=False
                )
            )
        else:
            print("Adding all slides...")
            indices_to_keep += slide_indices
        df_to_keep = df.loc[indices_to_keep, :].copy().reset_index(drop=True)

    print_stats(df_to_keep)
    df_to_keep.to_csv(os.path.join(output_root, "metadata.csv"))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    generate_final_metadata(slide_root=args.slide_root, output_root=args.output_root)
