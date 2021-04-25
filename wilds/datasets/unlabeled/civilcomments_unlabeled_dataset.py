import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import pandas as pd
import numpy as np

from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
from wilds.common.utils import map_to_id_array


class CivilCommentsUnlabeledDataset(WILDSUnlabeledDataset):
    """
    Unlabeled CivilComments-wilds toxicity classification dataset.
    This is a modified version of the original CivilComments dataset.

    Supported `split_scheme`:
        'official'

    Input (x):
        A comment on an online article, comprising one or more sentences of text.

    Website:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

    Original publication:
        @inproceedings{borkan2019nuanced,
          title={Nuanced metrics for measuring unintended bias with real data for text classification},
          author={Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain, Nithum and Vasserman, Lucy},
          booktitle={Companion Proceedings of The 2019 World Wide Web Conference},
          pages={491--500},
          year={2019}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    _NOT_IN_DATASET: int = -1

    _dataset_name: str = "civilcomments_unlabeled"
    _versions_dict: Dict[str, Dict[str, Union[str, int]]] = {
        "1.0": {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xde2797ca1e324b6586e19250d6df2f84/contents/blob/',
            'compressed_size': 90_644_480,
            "equivalent_dataset": "civilcomments_v2.0",
        },
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'unlabeled_data_with_identities.csv'),
            index_col=0)

        # Extract text
        self._text_array = list(self._metadata_df['comment_text'])

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # metadata_df contains split names in strings, so convert them to ints
        self._split_dict = { "test_unlabeled": 12 }
        self._split_names = { "test_unlabeled": "Unlabeled Test" }
        self._metadata_df['split'] = self.split_dict["test_unlabeled"]
        self._split_array = self._metadata_df['split'].values

        # Metadata (Not Available)
        self._metadata_fields = []
        self._metadata_array = torch.empty(len(self._metadata_df), 0)
        self._metdata_map = {}

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._text_array[idx]

