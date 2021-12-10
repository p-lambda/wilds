import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import pandas as pd
import numpy as np

from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
from wilds.common.utils import map_to_id_array


class AmazonUnlabeledDataset(WILDSUnlabeledDataset):
    """
    Unlabeled Amazon-WILDS dataset.
    This is a modified version of the 2018 Amazon Reviews dataset.

    Supported `split_scheme`:
        'official': official split, which is equivalent to 'user'
        'user': shifts to unseen reviewers
        'time': shifts from reviews written before 2013 to reviews written after 2013
        'category_subpopulation': the training distribution is a random subset following the natural distribution, and the
                                  evaluation splits include each category uniformly (to the extent it is possible)
        '*_generalization': domain generalization setting where the domains are categories. train categories vary.
        '*_baseline': oracle baseline splits for user or time shifts

    Input (x):
        Review text of maximum token length of 512.

    Metadata:
        reviewer: reviewer ID
        year: year in which the review was written
        category: product category
        product: product ID

    Website:
        https://nijianmo.github.io/amazon/index.html

    Original publication:
        @inproceedings{ni2019justifying,
          author = {J. Ni and J. Li and J. McAuley},
          booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
          pages = {188--197},
          title = {Justifying recommendations using distantly-labeled reviews and fine-grained aspects},
          year = {2019},
        }

    License:
        None. However, the original authors request that the data be used for research purposes only.
    """

    _NOT_IN_DATASET: int = -1

    _dataset_name: str = "amazon_unlabeled"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0xe3ed909786d34ee79d430d065582aa29/contents/blob/",
            "compressed_size": 1_989_805_589,
            "equivalent_dataset": "amazon_v2.1",
        },
    }

    def __init__(
        self,
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official",
    ):
        # Dataset information
        self._version: Optional[str] = version
        # The official split is to split by users
        self._split_scheme: str = "user" if split_scheme == "official" else split_scheme
        # Path of the dataset
        self._data_dir: str = self.initialize_data_dir(root_dir, download)

        # Load data
        data_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_dir, "reviews.csv"),
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
        split_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_dir, "splits", f"{self.split_scheme}.csv")
        )
        is_in_dataset: bool = (
            split_df["split"] != AmazonUnlabeledDataset._NOT_IN_DATASET
        )
        split_df = split_df[is_in_dataset]
        data_df = data_df[is_in_dataset]
        # Get arrays
        self._split_array: List[str] = split_df["split"].values
        self._input_array: List[str] = list(data_df["reviewText"])
        # Get metadata
        (
            self._metadata_fields,
            self._metadata_array,
            self._metadata_map,
        ) = self.load_metadata(data_df, self.split_array)
        # Get y from metadata
        self._y_type: str = "long"
        self._y_array = getattr(
            self.metadata_array[:, self.metadata_fields.index("y")], self._y_type
        )()
        # Set split info
        self.initialize_split_dicts()

        super().__init__(root_dir, download, self._split_scheme)

    def get_input(self, idx) -> str:
        return self._input_array[idx]

    def initialize_split_dicts(self):
        if self.split_scheme == "user":
            self._split_dict = {
                "val_unlabeled": 11,
                "test_unlabeled": 12,
                "extra_unlabeled": 13,
            }
            self._split_names = {
                "val_unlabeled": "Unlabeled Validation",
                "test_unlabeled": "Unlabeled Test",
                "extra_unlabeled": "Unlabeled Extra",
            }
        else:
            raise ValueError(f"Split scheme {self.split_scheme} is not recognized.")

    def load_metadata(
        self, data_df, split_array
    ) -> Tuple[List[str], torch.Tensor, Dict]:
        # Get metadata
        columns: List[str] = ["reviewerID", "asin", "category", "reviewYear", "overall"]
        metadata_fields: List[str] = ["user", "product", "category", "year", "y"]
        metadata_df: pd.DataFrame = data_df[columns].copy()
        metadata_df.columns = metadata_fields

        sort_idx = np.argsort(split_array)
        ordered_maps = {}
        for field in ["user", "product", "category"]:
            # map to IDs in the order of split values
            ordered_maps[field] = pd.unique(metadata_df.iloc[sort_idx][field])
        ordered_maps["y"] = range(1, 6)
        ordered_maps["year"] = range(
            metadata_df["year"].min(), metadata_df["year"].max() + 1
        )
        metadata_map, metadata = map_to_id_array(metadata_df, ordered_maps)
        return metadata_fields, torch.from_numpy(metadata.astype("long")), metadata_map
