from datetime import datetime
import os
from pathlib import Path

from PIL import Image
import pandas as pd
import numpy as np
import torch

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1


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
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xc01e117bb4504f988700408eaeeb16a8/contents/blob/',
            'compressed_size': 7_413_123_845}
    }

    def __init__(self, version=None, root_dir='rxrx1-wilds', download=False,
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
        split_dict = {'train': 0, 'test': 1}
        self._split_array = df.dataset.apply(split_dict.get).values

        # Filenames
        def create_filepath(row):
            filepath = os.path.join(row.experiment,
                                    f'Plate{row.plate}',
                                    f'{row.well}_s{row.site}.png')
            return filepath
        self._input_array = df.apply(create_filepath, axis=1).values

        # Labels
        self._y_array = torch.tensor(df['sirna_id'].values)
        self._n_classes = max(df['sirna_id']) + 1
        self._y_size = 1
        assert len(np.unique(df['sirna_id'])) == self._n_classes

        # Location/group info
        # FIXME need to enumerate experiments
        # n_groups = max(df['location_remapped']) + 1
        # self._n_groups = n_groups
        # assert len(np.unique(df['location_remapped'])) == self._n_groups

        # FIXME experiment and well are strings
        self._metadata_array = torch.tensor(
            np.stack([df['experiment'].values,
                      df['plate'].values,
                      df['well'].values,
                      df['site'].values,
                      self.y_array], axis=1)
        )
        self._metadata_fields = ['experiment', 'plate', 'well', 'site', 'y']

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['experiment'])
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
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
                        })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
        )

        return results, results_str

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
