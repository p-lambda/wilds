import argparse
import os
import pdb

import pandas as pd
import numpy as np

# Fix the seed for reproducibility
np.random.seed(0)

"""
Generate a CSV with the metadata for DomainNet (SENTRY version):

    @inproceedings{peng2019moment,
      title={Moment matching for multi-source domain adaptation},
      author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      pages={1406--1415},
      year={2019}
    }
    
    @article{prabhu2020sentry
       author = {Prabhu, Viraj and Khare, Shivam and Kartik, Deeksha and Hoffman, Judy},
       title = {SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation},
       year = {2020},
       journal = {arXiv preprint: 2012.11460},
    }
        
The dataset can be downloaded from http://ai.bu.edu/M3SDA.
The SENTRY splits can be found https://github.com/virajprabhu/SENTRY/tree/main/data/DomainNet/txt.
    
There are 586,576 images in 345 categories (airplane, ball, cup, etc.) across 6 domains (clipart, infograph, 
painting, quickdraw, real and sketch) in the original DomainNet dataset. Images are either PNG or JPG files.

The SENTRY version of the dataset has 40 categories across 4 domains:
"Due to labeling noise prevalent in the full version of DomainNet, we instead use the subset proposed in 
Tan et al. [42], which uses 40-commonly seen classes from four domains: Real (R), Clipart (C), Painting (P), 
and Sketch (S)."

The metadata CSV file has the following fields:

1. image_path: Path to the image file. The path has the following format: <domain>/<category>/<file_name>.
2. domain: One of the 4 possible domains.
3. split: One of "train" or "test".
4. category: One of the 40 possible categories.
5. y: Given to us by the SENTRY split  

Example usage:

    python dataset_preprocessing/domainnet/generate_sentry_metadata.py <path to Sentry splits>.

"""

DOMAINS = ["clipart", "painting", "real", "sketch"]
METADATA_COLUMNS = ["image_path", "domain", "split", "category", "y"]
NUM_OF_CATEGORIES = 40
TEST_SPLIT = "test"
TRAIN_SPLIT = "train"


def main(sentry_splits_path):
    def process_split(split, split_path):
        count = 0
        categories = set()
        with open(split_path) as f:
            for line in f.readlines():
                image_path, label = line.strip().split(" ")
                metadata_values = image_path.split(os.path.sep)
                metadata_dict["image_path"].append(image_path)
                metadata_dict["domain"].append(metadata_values[0])
                metadata_dict["split"].append(split)
                metadata_dict["category"].append(metadata_values[1])
                categories.add(metadata_values[1])
                metadata_dict["y"].append(int(label))
                count += 1
        assert len(categories) == NUM_OF_CATEGORIES
        return count

    print("Generating sentry_metadata.csv for DomainNet (SENTRY version)...")

    metadata_dict = {column: [] for column in METADATA_COLUMNS}
    for domain in DOMAINS:
        train_count = process_split(
            TRAIN_SPLIT,
            os.path.join(sentry_splits_path, f"{domain}_{TRAIN_SPLIT}_mini.txt"),
        )
        test_count = process_split(
            TEST_SPLIT,
            os.path.join(sentry_splits_path, f"{domain}_{TEST_SPLIT}_mini.txt"),
        )
        total_count = train_count + test_count
        train_percentage = np.round(float(train_count) / total_count * 100.0, 2)
        test_percentage = np.round(float(test_count) / total_count * 100.0, 2)
        print(
            f"Domain {domain} had {train_count} ({train_percentage}%) training examples "
            f"and {test_count} ({test_percentage}%) test examples with a total of {total_count} examples."
        )

    # Write metadata out as a CSV file
    metadata_df = pd.DataFrame(metadata_dict)
    metadata_path = os.path.join(sentry_splits_path, "sentry_metadata.csv")
    print(f"Writing metadata out to {metadata_path}...")
    metadata_df.to_csv(metadata_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a CSV with the metadata for DomainNet (SENTRY version)."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the DomainNet dataset downloaded from http://ai.bu.edu/M3SDA",
    )

    args = parser.parse_args()
    main(args.path)
