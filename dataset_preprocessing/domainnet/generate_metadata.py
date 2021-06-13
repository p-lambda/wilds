import argparse
import os
import pdb

import pandas as pd
import numpy as np

# Fix the seed for reproducibility
np.random.seed(0)

"""
Generate a CSV with the metadata for DomainNet:

    @inproceedings{peng2019moment,
      title={Moment matching for multi-source domain adaptation},
      author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      pages={1406--1415},
      year={2019}
    }
    
There are 586,576 images in 345 categories (airplane, ball, cup, etc.) across 6 domains (clipart, infograph, 
painting, quickdraw, real and sketch). Images are either PNG or JPG files.

The metadata CSV file has the following fields:

1. image_path: Path to the image file. The path has the following format: <domain>/<category>/<file_name>.
2. domain: One of the 6 possible domains.
3. split: One of "train", "val" or "test".
4. category: One of the 345 possible categories.
5. y: The index corresponding to the category (e.g. 1 if the image is of an aircraft carrier). 

Example usage:

    python dataset_preprocessing/domainnet/generate_metadata.py ../data/domainnet_v1.0

"""

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
METADATA_COLUMNS = ["image_path", "domain", "split", "category", "y"]
NUM_OF_CATEGORIES = 345


def main(dataset_path, split_val=False):
    print("Generating metadata.csv for DomainNet...")

    # Build mapping of image to split ("train", "val" or "test) and label
    categories = [""] * NUM_OF_CATEGORIES
    image_info = dict()
    original_splits_path = os.path.join(dataset_path, "original_splits")
    for domain in DOMAINS:
        train_count = 0
        with open(os.path.join(original_splits_path, f"{domain}_train.txt")) as f:
            for line in f.readlines():
                image_path, label = line.strip().split(" ")
                categories[int(label)] = image_path.split(os.path.sep)[1]
                image_info[image_path] = ["train", label]
                train_count += 1

        test_count = 0
        bucketed_test_set = [[] for _ in range(NUM_OF_CATEGORIES)]
        with open(os.path.join(original_splits_path, f"{domain}_test.txt")) as f:
            for line in f.readlines():
                image_path, label = line.strip().split(" ")
                label = int(label)
                image_info[image_path] = ["test", label]
                test_count += 1
                bucketed_test_set[label].append(image_path)

        total_count = train_count + test_count
        train_percentage = np.round(float(train_count) / total_count * 100.0, 2)
        test_percentage = np.round(float(test_count) / total_count * 100.0, 2)
        print(
            f"Domain {domain} originally had {train_count} ({train_percentage}%) training examples "
            f"and {test_count} ({test_percentage}%) test examples with a total of {total_count} examples."
        )

        val_count = 0
        if split_val:
            # Go from 70-30 train-test split to 70-15-15 train-val-test split
            print("Creating a validation set from the existing test set...")
            for category_images in bucketed_test_set:
                new_val_images = np.random.choice(
                    category_images, len(category_images) // 2, replace=False
                )
                for image_path in new_val_images:
                    image_info[image_path][0] = "val"
                    val_count += 1

        val_percentage = np.round(float(val_count) / total_count * 100.0, 2)
        test_count -= val_count
        test_percentage = np.round(float(test_count) / total_count * 100.0, 2)
        print(
            f"Domain {domain} now has {train_count} ({train_percentage}%) training examples, "
            f"{val_count} ({val_percentage}%) validation examples and {test_count} ({test_percentage}%) test "
            f"examples with a total of {total_count} examples.\n"
        )

    # For debugging
    print(f"Categories in order: {categories}")

    # Build metadata
    metadata_dict = {column: [] for column in METADATA_COLUMNS}
    for domain in DOMAINS:
        domain_path = os.path.join(dataset_path, domain)

        for category in os.listdir(domain_path):
            category_path = os.path.join(domain_path, category)
            if not os.path.isdir(category_path):
                continue

            for image in os.listdir(category_path):
                image_path = os.path.join(domain, category, image)
                if (
                    image.endswith(".jpg") or image.endswith(".png")
                ) and image_path in image_info:
                    split, y = image_info[image_path]
                    metadata_dict["image_path"].append(image_path)
                    metadata_dict["domain"].append(domain)
                    metadata_dict["split"].append(split)
                    metadata_dict["category"].append(category)
                    metadata_dict["y"].append(y)

    # Write metadata out as a CSV file
    metadata_df = pd.DataFrame(metadata_dict)
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    print(f"Writing metadata out to {metadata_path}...")
    metadata_df.to_csv(metadata_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a CSV with the metadata for DomainNet."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the DomainNet dataset downloaded from http://ai.bu.edu/M3SDA",
    )
    parser.add_argument(
        "--split-val",
        action="store_true",
        help="Whether to create a separate validation by splitting the existing test split "
        "in half (defaults to false).",
    )

    args = parser.parse_args()
    main(args.path, args.split_val)
