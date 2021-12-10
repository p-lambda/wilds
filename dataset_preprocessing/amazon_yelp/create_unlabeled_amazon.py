import argparse
import csv
import os

import numpy as np
import pandas as pd

# Fix the seed for reproducibility
np.random.seed(0)

"""
Create unlabeled splits for Amazon.

Usage:
    python dataset_preprocessing/amazon_yelp/create_unlabeled_amazon.py <path>
"""

NOT_IN_DATASET = -1

# Splits
# 'train': 0, 'val': 1, 'id_val': 2, 'test': 3, 'id_test': 4,
# 'val_unlabeled': 11, 'test_unlabeled': 12, 'extra_unlabeled': 13
(
    TRAIN,
    OOD_VAL,
    ID_VAL,
    OOD_TEST,
    ID_TEST,
) = range(5)
VAL_UNLABELED, TEST_UNLABELED, EXTRA_UNLABELED = range(11, 14)


def main(dataset_path):
    def output_split_sizes():
        print("-" * 50)
        print(f'Train size: {len(split_df[split_df["split"] == TRAIN])}')
        print(f'Val size: {len(split_df[split_df["split"] == OOD_VAL])}')
        print(f'ID Val size: {len(split_df[split_df["split"] == ID_VAL])}')
        print(f'Test size: {len(split_df[split_df["split"] == OOD_TEST])}')
        print(f'ID Test size: {len(split_df[split_df["split"] == ID_TEST])}')
        print(
            f'OOD Val Unlabeled size: {len(split_df[split_df["split"] == VAL_UNLABELED])}'
        )
        print(
            f'OOD Test Unlabeled size: {len(split_df[split_df["split"] == TEST_UNLABELED])}'
        )
        print(
            f'Extra Unlabeled size: {len(split_df[split_df["split"] == EXTRA_UNLABELED])}'
        )
        print(
            f'Number of examples not included: {len(split_df[split_df["split"] == NOT_IN_DATASET])}'
        )
        print(f'Number of unclean reviews: {len(split_df[~split_df["clean"]])}')
        print("-" * 50)
        print("\n")

    def set_unlabeled_split(split, reviewers):
        # Get unused reviews written by users from `reviewers`
        split_df.loc[
            (split_df["split"] == NOT_IN_DATASET)
            & split_df["clean"]
            & data_df["reviewerID"].isin(reviewers),
            "split",
        ] = split

    def validate_split(split, expected_reviewers_count):
        # Sanity check:
        # Ensure the number of reviewers equals the number of reviewers in its unlabeled counterpart
        # and each reviewer has at least 75 reviews.
        actual_reviewers_counts = (
            data_df[(split_df["split"] == split)]["reviewerID"].unique().size
        )
        assert (
            actual_reviewers_counts == expected_reviewers_count
        ), "The number of reviewers ({}) did not equal {}".format(
            actual_reviewers_counts, expected_reviewers_count
        )
        min_reviewers_count = (
            data_df[(split_df["split"] == split)]["reviewerID"].value_counts().min()
        )
        assert (
            min_reviewers_count >= 75
        ), "Each reviewer should have at least 75 reviews, but got a minimum of {} reviews.".format(
            min_reviewers_count
        )

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

    ood_val_reviewers_ids = data_df[
        split_df["split"] == OOD_VAL
    ].reviewerID.unique()  # 1334 users
    set_unlabeled_split(VAL_UNLABELED, ood_val_reviewers_ids)

    ood_test_reviewers_ids = data_df[
        split_df["split"] == OOD_TEST
    ].reviewerID.unique()  # 1334 users
    set_unlabeled_split(TEST_UNLABELED, ood_test_reviewers_ids)

    # For EXTRA_UNLABELED, use any users not in any of the other splits
    existing_reviewer_ids = np.concatenate(
        [
            ood_test_reviewers_ids,
            ood_val_reviewers_ids,
            data_df[split_df["split"] == TRAIN].reviewerID.unique(),
            data_df[split_df["split"] == ID_VAL].reviewerID.unique(),
            data_df[split_df["split"] == ID_TEST].reviewerID.unique(),
        ]
    )
    # There are 151,736 extra reviewers
    extra_reviewers_ids = data_df[
        ~data_df.reviewerID.isin(existing_reviewer_ids)
    ].reviewerID.unique()
    set_unlabeled_split(EXTRA_UNLABELED, extra_reviewers_ids)

    # Exclude reviewers with less than 75 reviews.
    review_counts = data_df[(split_df["split"] == EXTRA_UNLABELED)][
        "reviewerID"
    ].value_counts()
    reviewers_to_filter_out = review_counts[review_counts < 75].keys()
    split_df.loc[
        (split_df["split"] == EXTRA_UNLABELED)
        & data_df["reviewerID"].isin(reviewers_to_filter_out),
        "split",
    ] = NOT_IN_DATASET

    # We are done splitting, output stats.
    output_split_sizes()

    # Sanity checks
    validate_split(VAL_UNLABELED, ood_val_reviewers_ids.size)
    validate_split(TEST_UNLABELED, ood_test_reviewers_ids.size)
    # After filtering out unclean reviews and ensuring >= 75 reviews per reviewer, we are left with 21,694 reviewers.
    validate_split(EXTRA_UNLABELED, 21694)

    # Write out the new unlabeled split to user.csv
    split_df.to_csv(user_csv_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create unlabeled splits for Amazon.")
    parser.add_argument(
        "path",
        type=str,
        help="Path to the Amazon dataset",
    )
    args = parser.parse_args()
    main(args.path)
