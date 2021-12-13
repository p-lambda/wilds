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
    Unlabeled CivilComments-WILDS toxicity classification dataset.
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
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x1c471f23448e4518b000fe47aa7724e0/contents/blob/',
            'compressed_size': 254_142_009
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
        self._split_dict = { "extra_unlabeled": 13 }
        self._split_names = { "extra_unlabeled": "Unlabeled Extra" }
        self._metadata_df['split'] = self.split_dict["extra_unlabeled"]
        self._split_array = self._metadata_df['split'].values

        # Metadata (Not Available)
        # We want grouper to assign all values to their own group, so fill
        # all metadata fields with '2'. The normal dataset has binary metadata,
        # so this will not overlap.
        self._identity_vars = [
            'male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white'
        ]
        self._auxiliary_vars = [
            'identity_any',
            'severe_toxicity',
            'obscene',
            'threat',
            'insult',
            'identity_attack',
            'sexual_explicit'
        ]

        self._y_array = torch.LongTensor(self._metadata_df['toxicity'].values >= 0.5)
        self._metadata_array = torch.cat(
            (
                torch.ones(
                    len(self._metadata_df),
                    len(self._identity_vars) + len(self._auxiliary_vars)
                ) * 2,
                self._y_array.unsqueeze(dim=-1)
            ),
            axis=1
        )
        self._metadata_fields = self._identity_vars + self._auxiliary_vars + ['y']

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._text_array[idx]

