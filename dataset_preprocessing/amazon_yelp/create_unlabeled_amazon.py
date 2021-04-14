import argparse
import csv
import os
import pdb

import numpy as np
import pandas as pd

# Fix the seed for reproducibility
np.random.seed(0)

"""
Create unlabeled split for the Amazon dataset.

Usage:
    python dataset_preprocessing/amazon_yelp/create_unlabeled_amazon.py <path>
"""

NOT_IN_DATASET = -1
# Split: {'train': 0, 'val': 1, 'id_val': 2, 'test': 3, 'id_test': 4, 'ood_unlabeled': 5}
TRAIN, OOD_VAL, ID_VAL, OOD_TEST, ID_TEST, OOD_UNLABELED = range(6)


def main(dataset_path):
    def output_split_sizes():
        print("-" * 50)
        print(f'Train size: {len(split_df[split_df["split"] == TRAIN])}')
        print(f'Val size: {len(split_df[split_df["split"] == OOD_VAL])}')
        print(f'ID Val size: {len(split_df[split_df["split"] == ID_VAL])}')
        print(f'Test size: {len(split_df[split_df["split"] == OOD_TEST])}')
        print(f'ID Test size: {len(split_df[split_df["split"] == ID_TEST])}')
        print(
            f'OOD Unlabeled size: {len(split_df[split_df["split"] == OOD_UNLABELED])}'
        )
        print(
            f'Number of examples not included: {len(split_df[split_df["split"] == NOT_IN_DATASET])}'
        )
        print("-" * 50)
        print("\n")

    data_df = pd.read_csv(
        os.path.join(dataset_path, "reviews.csv"),
        dtype={
            "reviewerID": str,
            "asin": str,
            "reviewTime": str,
            "unixReviewTime": int,
            "reviewText": str,
            "summary": str,
            "verified": bool,
            "category": str,
            "reviewYear": int,
        },
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONNUMERIC,
    )
    user_csv_path = os.path.join(dataset_path, "splits", "user.csv")
    split_df = pd.read_csv(user_csv_path)
    assert split_df.shape[0] == data_df.shape[0]
    output_split_sizes()

    # Get unused reviews written by users from the OOD test set:
    ood_test_data_df = data_df[split_df["split"] == OOD_TEST]
    ood_test_reviewers_ids = ood_test_data_df.reviewerID.unique()
    split_df.loc[
        (split_df["split"] == NOT_IN_DATASET)
        & split_df["clean"]
        & data_df["reviewerID"].isin(ood_test_reviewers_ids),
        "split",
    ] = OOD_UNLABELED
    output_split_sizes()

    # Sanity check - Ensure there are 1334 reviewers and each reviewer should have at least 75 reviews
    assert (
        data_df[(split_df["split"] == OOD_UNLABELED)]["reviewerID"].unique().size
        == ood_test_reviewers_ids.size
    )
    assert (
        data_df[(split_df["split"] == OOD_UNLABELED)]["reviewerID"].value_counts().min()
        == 75
    )

    # Write out the new unlabeled split to user.csv
    split_df.to_csv(user_csv_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create unlabeled split for the Amazon dataset."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the Amazon dataset",
    )

    args = parser.parse_args()
    main(args.path)
