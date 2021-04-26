import argparse
import csv
import os
import pdb

import numpy as np
import pandas as pd

# Fix the seed for reproducibility
np.random.seed(0)

"""
Process unlabeled data in CivilComments
"""

TRAIN, VAL, TEST, UNLABELED = ('train', 'val', 'test', 'test_unlabeled')

def load_unlabeled_df(root):
    """
    Loads the raw data where we don't have identity annotations.
    """
    df = pd.read_csv(os.path.join(root, 'all_data.csv'))
    df = df.loc[(df['identity_annotator_count'] == 0), :]
    df = df.reset_index(drop=True)
    return df

def load_labeled_df(root):
    """
    Loads the processed data for which we do have identity annotations.
    """
    df = pd.read_csv(os.path.join(root, 'all_data_with_identities.csv'), index_col=0)
    return df

def merge_dfs(unlabeled, labeled):
    """
    Drops columns that are in unlabeled but not labeled
    Adds columns that are in labeled but not unlabeled and sets values to NaN
    """
    common_cols = unlabeled.columns & labeled.columns
    unlabeled = unlabeled[common_cols]
    joint = labeled.append(unlabeled, ignore_index = True)
    return joint

def main(args):
    unlabeled = load_unlabeled_df(args.root)
    labeled = load_labeled_df(args.root)

    # set all unlabeled examples to the same split
    unlabeled['split'] = UNLABELED

    # merge unlabeled, labeled dfs
    joint = merge_dfs(unlabeled, labeled)
    assert (joint.columns == labeled.columns).all()

    def output_split_sizes(df):
        print("-" * 50)
        print(f'Train size: {len(df[df["split"] == TRAIN])}')
        print(f'Val size: {len(df[df["split"] == VAL])}')
        print(f'Test size: {len(df[df["split"] == TEST])}')
        print(
            f'Unlabeled size: {len(df[df["split"] == UNLABELED])}'
        )
        print("-" * 50)
        print("\n")

    output_split_sizes(joint)
    
    # Write out the new unlabeled split to user.csv
    joint.to_csv(f'{args.root}/all_data_with_identities_and_unlabeled.csv', index=True)
    joint[joint['split'] == UNLABELED].to_csv(f'{args.root}/unlabeled_data_with_identities.csv', index=True)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create unlabeled splits for CivilComments.")
    parser.add_argument(
        "--root",
        type=str,
        help="Path to the dir containing the CivilComments processed labeled csv and full csv.",
    )
    args = parser.parse_args()
    main(args)
