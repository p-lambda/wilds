import argparse
import csv
import os

import pandas as pd
import numpy as np

# Fix the seed for reproducibility
np.random.seed(0)

"""
Subsample the Amazon dataset.

Usage:
    python dataset_preprocessing/amazon_yelp/subsample_amazon.py <path> <frac>
"""

NOT_IN_DATASET = -1
# Split: {'train': 0, 'val': 1, 'id_val': 2, 'test': 3, 'id_test': 4}
TRAIN, OOD_VAL, ID_VAL, OOD_TEST, ID_TEST = range(5)


def main(dataset_path, frac=0.25):
    def output_dataset_sizes(split_df):
        print("-" * 50)
        print(f'Train size: {len(split_df[split_df["split"] == TRAIN])}')
        print(f'Val size: {len(split_df[split_df["split"] == OOD_VAL])}')
        print(f'ID Val size: {len(split_df[split_df["split"] == ID_VAL])}')
        print(f'Test size: {len(split_df[split_df["split"] == OOD_TEST])}')
        print(f'ID Test size: {len(split_df[split_df["split"] == ID_TEST])}')
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
    output_dataset_sizes(split_df)

    train_data_df = data_df[split_df["split"] == 0]
    train_reviewer_ids = train_data_df.reviewerID.unique()
    print(f"Number of unique reviewers in train set: {len(train_reviewer_ids)}")

    # Randomly sample (1 - frac) x number of reviewers
    # Blackout all the reviews belonging to the randomly sampled reviewers
    subsampled_reviewers_count = int((1 - frac) * len(train_reviewer_ids))
    subsampled_reviewers = np.random.choice(
        train_reviewer_ids, subsampled_reviewers_count, replace=False
    )
    print(subsampled_reviewers)

    blackout_indices = train_data_df[
        train_data_df["reviewerID"].isin(subsampled_reviewers)
    ].index

    # Mark all the corresponding reviews of blackout_indices as -1
    split_df.loc[blackout_indices, "split"] = NOT_IN_DATASET
    output_dataset_sizes(split_df)

    # Mark duplicates
    duplicated_within_user = data_df[["reviewerID", "reviewText"]].duplicated()
    df_deduplicated_within_user = data_df[~duplicated_within_user]
    duplicated_text = df_deduplicated_within_user[
        df_deduplicated_within_user["reviewText"]
        .apply(lambda x: x.lower())
        .duplicated(keep=False)
    ]["reviewText"]
    duplicated_text = set(duplicated_text.values)
    data_df["duplicate"] = (
        data_df["reviewText"].isin(duplicated_text)
    ) | duplicated_within_user

    # Mark html candidates
    data_df["contains_html"] = data_df["reviewText"].apply(
        lambda x: "<" in x and ">" in x
    )

    # Mark clean ones
    data_df["clean"] = ~data_df["duplicate"] & ~data_df["contains_html"]

    # Clear ID val and ID test since we're regenerating
    split_df.loc[split_df["split"] == ID_VAL, "split"] = NOT_IN_DATASET
    split_df.loc[split_df["split"] == ID_TEST, "split"] = NOT_IN_DATASET

    # Regenerate ID val and ID test
    train_reviewer_ids = data_df[split_df["split"] == TRAIN]["reviewerID"].unique()
    np.random.shuffle(train_reviewer_ids)
    cutoff = int(len(train_reviewer_ids) / 2)
    id_val_reviewer_ids = train_reviewer_ids[:cutoff]
    id_test_reviewer_ids = train_reviewer_ids[cutoff:]
    split_df.loc[
        (split_df["split"] == NOT_IN_DATASET)
        & data_df["clean"]
        & data_df["reviewerID"].isin(id_val_reviewer_ids),
        "split",
    ] = ID_VAL
    split_df.loc[
        (split_df["split"] == NOT_IN_DATASET)
        & data_df["clean"]
        & data_df["reviewerID"].isin(id_test_reviewer_ids),
        "split",
    ] = ID_TEST

    # Sanity check
    assert (
        data_df[(split_df["split"] == ID_VAL)]["reviewerID"].value_counts().min() == 75
    )
    assert (
        data_df[(split_df["split"] == ID_VAL)]["reviewerID"].value_counts().max() == 75
    )
    assert (
        data_df[(split_df["split"] == ID_TEST)]["reviewerID"].value_counts().min() == 75
    )
    assert (
        data_df[(split_df["split"] == ID_TEST)]["reviewerID"].value_counts().max() == 75
    )

    # Write out the new splits to user.csv
    output_dataset_sizes(split_df)
    split_df.to_csv(user_csv_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample the Amazon dataset.")
    parser.add_argument(
        "path",
        type=str,
        help="Path to the Amazon dataset",
    )
    parser.add_argument(
        "frac",
        type=float,
        help="Subsample fraction",
    )

    args = parser.parse_args()
    main(args.path, args.frac)
