import csv
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import pandas as pd
from PIL import Image

from wilds.common.utils import map_to_id_array
from wilds.datasets.domainnet_dataset import DOMAIN_NET_CATEGORIES, DOMAIN_NET_DOMAINS, SENTRY_DOMAINS
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset


class DomainNetUnlabeledDataset(WILDSUnlabeledDataset):
    """
    Unlabeled DomainNet dataset.

    Supported `split_scheme`:
        'official': use the official split from DomainNet

    Input (x):
        224 x 224 x 3 RGB image.

    Label (y):
        y is one of the 345 categories in DomainNet.

    Metadata:
        None

    Website:
        http://ai.bu.edu/M3SDA

    Original publication:

        @inproceedings{peng2019moment,
          title={Moment matching for multi-source domain adaptation},
          author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
          booktitle={Proceedings of the IEEE International Conference on Computer Vision},
          pages={1406--1415},
          year={2019}
        }

    SENTRY publication:

        @article{prabhu2020sentry
           author = {Prabhu, Viraj and Khare, Shivam and Kartik, Deeksha and Hoffman, Judy},
           title = {SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation},
           year = {2020},
           journal = {arXiv preprint: 2012.11460},
        }

    Fair Use Notice:
        "This dataset contains some copyrighted material whose use has not been specifically authorized by the copyright owners.
        In an effort to advance scientific research, we make this material available for academic research. We believe this
        constitutes a fair use of any such copyrighted material as provided for in section 107 of the US Copyright Law.
        In accordance with Title 17 U.S.C. Section 107, the material on this site is distributed without profit for
        non-commercial research and educational purposes. For more information on fair use please click here. If you wish
        to use copyrighted material on this site or in our dataset for purposes of your own that go beyond non-commercial
        research and academic purposes, you must obtain permission directly from the copyright owner."
    """

    _dataset_name: str = "domainnet_unlabeled"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x0b8ca76eef384b98b879d0c8c4681a32/contents/blob/",
            "compressed_size": 19_255_770_459,
            "equivalent_dataset": "domainnet_v1.0",
        },
    }

    def __init__(
        self,
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official",
        source_domain: str = "sketch",
        target_domain: str = "real",
        extra_domain: str = "clipart",
        use_sentry: bool = False,
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (224, 224)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = self.initialize_data_dir(root_dir, download)

        if use_sentry:
            for domain in [source_domain, target_domain, extra_domain]:
                assert domain in SENTRY_DOMAINS
            print("Using the SENTRY version of DomainNet (unlabeled)...")
            metadata_filename = "sentry_metadata.csv"
            self._n_classes = 40
        else:
            metadata_filename = "metadata.csv"
            self._n_classes = 345

        # Load data
        metadata_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_dir, metadata_filename),
            dtype={
                "image_path": str,
                "domain": str,
                "split": str,
                "category": str,
                "y": int,
            },
            keep_default_na=False,
            na_values=[],
            quoting=csv.QUOTE_NONNUMERIC,
        )
        target_metadata_df = metadata_df.loc[metadata_df["domain"] == target_domain]
        extra_metadata_df = metadata_df.loc[metadata_df["domain"] == extra_domain]
        metadata_df = pd.concat([target_metadata_df, extra_metadata_df])

        self._input_image_paths = metadata_df["image_path"].values
        self._y_array = torch.from_numpy(metadata_df["y"].values).type(torch.LongTensor)
        self.initialize_split_dicts()
        self.initialize_split_array(metadata_df, target_domain, extra_domain)

        # Populate metadata fields
        self._metadata_fields = ["domain", "category", "y"]
        metadata_df = metadata_df[self._metadata_fields]
        possible_metadata_values = {
            "domain": DOMAIN_NET_DOMAINS,
            "category": DOMAIN_NET_CATEGORIES,
            "y": range(self._n_classes),
        }
        self._metadata_map, metadata = map_to_id_array(
            metadata_df, possible_metadata_values
        )
        self._metadata_array = torch.from_numpy(metadata.astype("long"))

        super().__init__(root_dir, download, self._split_scheme)

    def get_input(self, idx) -> str:
        img_path = os.path.join(self.data_dir, self._input_image_paths[idx])
        img = Image.open(img_path).convert("RGB")
        return img

    def initialize_split_dicts(self):
        if self.split_scheme == "official":
            self._split_dict = {
                "test_unlabeled": 12,
                "extra_unlabeled": 13,
            }
            self._split_names = {
                "test_unlabeled": "Unlabeled Test",
                "extra_unlabeled": "Unlabeled Extra",
            }
        else:
            raise ValueError(f"Split scheme {self.split_scheme} is not recognized.")

    def initialize_split_array(self, metadata_df, target_domain, extra_domain):
        def get_split(row):
            if row["domain"] == target_domain:
                if row["split"] == "train":
                    return 12
                else:
                    return -1
            elif row["domain"] == extra_domain:
                if row["split"] == "train":
                    return 13
                else:
                    return -1
            else:
                raise ValueError(
                    f"Domain should be one of {target_domain}, {extra_domain}"
                )

        self._split_array = metadata_df.apply(
            lambda row: get_split(row), axis=1
        ).to_numpy()
