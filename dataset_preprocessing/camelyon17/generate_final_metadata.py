import pandas as pd
from matplotlib import pyplot as plt
import argparse
import os,sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def generate_final_metadata(output_root):
    df = pd.read_csv(os.path.join(output_root, 'all_patch_coords.csv'),
                     index_col=0,
                     dtype={
                        'patient': 'str',
                        'tumor': 'int'
                     })

    # Assign slide numbers to patients + nodes
    patient_node_list = list(set(df[['patient', 'node']].itertuples(index=False, name=None)))
    patient_node_list.sort()
    patient_node_to_slide_map = {}
    for idx, (patient, node) in enumerate(patient_node_list):
        patient_node_to_slide_map[(patient, node)] = idx

    for (patient, node), slide_idx in patient_node_to_slide_map.items():
        mask = (df['patient'] == patient) & (df['node'] == node)
        df.loc[mask, 'slide'] = slide_idx
    df['slide'] = df['slide'].astype('int')

    # The raw data has the following assignments:
    # Center 0: patients 0 to 19
    # Center 1: patients 20 to 39
    # Center 2: patients 40 to 59
    # Center 3: patients 60 to 79
    # Center 4: patients 80 to 99
    num_centers = 5
    patients_per_center = 20
    df['center'] = df['patient'].astype('int') // patients_per_center

    for k in range(num_centers):
        print(f"center {k}: "
              f"{np.sum((df['center'] == k) & (df['tumor'] == 0)):6d} non-tumor, "
              f"{np.sum((df['center'] == k) & (df['tumor'] == 1)):6d} tumor")

    for center, slide in set(df[['center', 'slide']].itertuples(index=False, name=None)):
        assert center == slide // 10

    # Keep all tumor patches, except if the slide has fewer normal than tumor patches
    # (slide 096 in center 4)
    # in which case we discard the excess tumor patches
    indices_to_keep = []
    np.random.seed(0)
    tumor_mask = df['tumor'] == 1
    for slide in set(df['slide']):
        slide_mask = (df['slide'] == slide)
        num_tumor = np.sum(slide_mask & tumor_mask)
        num_non_tumor = np.sum(slide_mask & ~tumor_mask)
        slide_indices_with_tumor = list(df.index[slide_mask & tumor_mask])
        indices_to_keep += list(np.random.choice(
            slide_indices_with_tumor,
            size=min(num_tumor, num_non_tumor),
            replace=False))

    tumor_keep_mask = np.zeros(len(df))
    tumor_keep_mask[df.index[indices_to_keep]] = 1

    # Within each center and split, keep same number of normal patches as tumor patches
    for center in range(num_centers):
        print(f'Center {center}:')
        center_mask = df['center'] == center
        num_tumor = np.sum(center_mask & tumor_keep_mask)
        print(f'  Num tumor: {num_tumor}')

        num_non_tumor = np.sum(center_mask & ~tumor_mask)
        center_indices_without_tumor = list(df.index[center_mask & ~tumor_mask])
        indices_to_keep += list(np.random.choice(
            center_indices_without_tumor,
            size=min(num_tumor, num_non_tumor),
            replace=False))

        print(f'  Num non-tumor: {min(num_tumor, num_non_tumor)} out of {num_non_tumor} ({min(num_tumor, num_non_tumor) / num_non_tumor * 100:.1f}%)')

        df_to_keep = df.loc[indices_to_keep, :].copy().reset_index(drop=True)

    val_frac = 0.1

    split_dict = {
        'train': 0,
        'val': 1,
        'test': 2
    }

    df_to_keep['split'] = split_dict['train']

    all_indices = list(df_to_keep.index)
    val_indices = list(np.random.choice(
        all_indices,
        size=int(val_frac * len(all_indices)),
        replace=False))
    df_to_keep.loc[val_indices, 'split'] = split_dict['val']

    print('Statistics by center:')
    for center in range(num_centers):
        tumor_mask = df_to_keep['tumor'] == 1
        center_mask = df_to_keep['center'] == center
        num_tumor = np.sum(center_mask & tumor_mask)
        num_non_tumor = np.sum(center_mask & ~tumor_mask)

        print(f'Center {center}')
        print(f'  {num_tumor} / {num_tumor + num_non_tumor} ({num_tumor / (num_tumor + num_non_tumor) * 100:.1f}%) tumor')

    df_to_keep.to_csv(os.path.join(output_root, 'metadata.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', required=True)
    args = parser.parse_args()
    generate_final_metadata(args.output_root)
