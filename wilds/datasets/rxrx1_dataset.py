import os
from pathlib import Path

from PIL import Image
import pandas as pd
import numpy as np
import torch

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy


class RxRx1Dataset(WILDSDataset):
    """
    The RxRx1 Dataset.
    This is a modified version of the original RxRx1 dataset.

    Input (x):
        3-channel fluorescent microscopy images of cells

    Label (y):
        y is one of 1,139 classes:
        - 0 to 1107: treatment siRNAs
        - 1108 to 1137: positive control siRNAs
        - 1138: negative control siRNA

    Metadata:
        Each image is annotated with its experiment, plate, well, and site, as
        well as with the id of the siRNA the cells were perturbed with.

    Website:
        https://www.rxrx.ai/rxrx1
        https://www.kaggle.com/c/recursion-cellular-image-classification

    FIXME
    Original publication:
        @article{,
            title={},
            author={},
            journal={},
            year={}
        }

    License:
        This work is licensed under a Creative Commons
        Attribution-NonCommercial-ShareAlike 4.0 International License. To view
        a copy of this license, visit
        http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
        Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
    """
    _dataset_name = 'rxrx1'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x6b7a05a3056a434498f0bb1252eb8440/contents/blob/',
            'compressed_size': 7_413_123_845}
    }

    def __init__(self, version=None, root_dir='data', download=False,
                 split_scheme='official'):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load splits
        df = pd.read_csv(self._data_dir / 'metadata.csv')

        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
        self._split_array = df.dataset.apply(self._split_dict.get).values

        # Filenames
        def create_filepath(row):
            filepath = os.path.join('images',
                                    row.experiment,
                                    f'Plate{row.plate}',
                                    f'{row.well}_s{row.site}.png')
            return filepath
        self._input_array = df.apply(create_filepath, axis=1).values

        # Labels
        self._y_array = torch.tensor(df['sirna_id'].values)
        self._n_classes = max(df['sirna_id']) + 1
        self._y_size = 1
        assert len(np.unique(df['sirna_id'])) == self._n_classes

        # Convert experiment and well from strings to idxs
        indexed_metadata = {}
        self._metadata_map = {}
        for key in ['cell_type', 'experiment', 'well']:
            all_values = list(df[key].unique())
            value_to_idx_map = {value: idx for idx, value in enumerate(all_values)}
            value_idxs = [value_to_idx_map[value] for value in df[key].tolist()]
            self._metadata_map[key] = all_values
            indexed_metadata[key] = value_idxs

        self._metadata_array = torch.tensor(
            np.stack([indexed_metadata['cell_type'],
                      indexed_metadata['experiment'],
                      df['plate'].values,
                      indexed_metadata['well'],
                      df['site'].values,
                      self.y_array], axis=1)
        )
        self._metadata_fields = ['cell_type', 'experiment', 'plate', 'well', 'site', 'y']

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['cell_type'])
        )

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are
                predicted labels (LongTensor). But they can also be other model
                outputs such that prediction_fn(y_pred) are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / self._input_array[idx]
        img = Image.open(img_path)

        return img
