import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import pandas as pd
import numpy as np

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.utils import map_to_id_array
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper


class AmazonDataset(WILDSDataset):
    """
    Amazon dataset.
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

    Label (y):
        y is the star rating (0,1,2,3,4 corresponding to 1-5 stars)

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

    _dataset_name: str = "amazon"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x60237058e01749cda7b0701c2bd01420/contents/blob/",
            "compressed_size": 4_066_541_568,
        },
        "2.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0xadbf6198d3a64bdc96fb64d6966b5e79/contents/blob/",
            "compressed_size": 1_987_523_759,
        },
        "2.1": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0xe3ed909786d34ee79d430d065582aa29/contents/blob/",
            "compressed_size": 1_989_805_589,
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
        self._y_type: str = "long"
        self._y_size: int = 1
        self._n_classes: int = 5  # One for each star rating
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
        is_in_dataset: bool = split_df["split"] != AmazonDataset._NOT_IN_DATASET
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
        self._y_array = getattr(
            self.metadata_array[:, self.metadata_fields.index("y")], self._y_type
        )()
        # Set split info
        self.initialize_split_dicts()
        # eval
        self.initialize_eval_grouper()
        super().__init__(root_dir, download, self._split_scheme)

    def get_input(self, idx) -> str:
        return self._input_array[idx]

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)

        if self.split_scheme == "user":
            # first compute groupwise accuracies
            g: torch.Tensor= self._eval_grouper.metadata_to_group(metadata)
            results: Dict[str, Any] = {
                **metric.compute(y_pred, y_true),
                **metric.compute_group_wise(
                    y_pred, y_true, g, self._eval_grouper.n_groups
                ),
            }

            accs: List[float] = []
            for group_idx in range(self._eval_grouper.n_groups):
                group_str: str = self._eval_grouper.group_field_str(group_idx)
                group_metric: float = results.pop(metric.group_metric_field(group_idx))
                group_counts: int = results.pop(metric.group_count_field(group_idx))
                results[f"{metric.name}_{group_str}"] = group_metric
                results[f"count_{group_str}"] = group_counts
                if group_counts > 0:
                    accs.append(group_metric)

            accs = np.array(accs)
            results["10th_percentile_acc"] = np.percentile(accs, 10)
            results[f"{metric.worst_group_metric_field}"] = metric.worst(accs)
            results_str = (
                f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
                f"10th percentile {metric.name}: {results['10th_percentile_acc']:.3f}\n"
                f"Worst-group {metric.name}: {results[metric.worst_group_metric_field]:.3f}\n"
            )
            return results, results_str
        else:
            return self.standard_group_eval(
                metric, self._eval_grouper, y_pred, y_true, metadata
            )

    def initialize_split_dicts(self):
        if self.split_scheme in ("user", "time") or self.split_scheme.endswith(
            "_generalization"
        ):
            # Category generalization
            self._split_dict: Dict[str, int] = {
                "train": 0,
                "val": 1,
                "id_val": 2,
                "test": 3,
                "id_test": 4,
            }
            self._split_names: Dict[str, str] = {
                "train": "Train",
                "val": "Validation (OOD)",
                "id_val": "Validation (ID)",
                "test": "Test (OOD)",
                "id_test": "Test (ID)",
            }
            self._source_domain_splits = [0, 2, 4]
        elif (
            self.split_scheme == "category_subpopulation"
            or self.split_scheme.endswith("_baseline")
        ):
            # Use defaults
            pass
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

    def initialize_eval_grouper(self):
        if self.split_scheme == "user":
            self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=["user"]
            )
        elif (
            self.split_scheme.endswith("generalization")
            or self.split_scheme == "category_subpopulation"
        ):
            self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=["category"]
            )
        elif self.split_scheme in ("time", "time_baseline"):
            self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=["year"]
            )
        elif self.split_scheme.endswith("_baseline"):  # user baselines
            self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=["user"]
            )
        else:
            raise ValueError(f"Split scheme {self.split_scheme} not recognized.")
