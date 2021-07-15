from datetime import datetime
from pathlib import Path
import os

from PIL import Image
import pandas as pd
import numpy as np
import torch
import json

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1


class IWildCamDataset(WILDSDataset):
    """
        The iWildCam2020 dataset.
        This is a modified version of the original iWildCam2020 competition dataset.
        Supported `split_scheme`:
            - 'official'
        Input (x):
            RGB images from camera traps
        Label (y):
            y is one of 186 classes corresponding to animal species
        Metadata:
            Each image is annotated with the ID of the location (camera trap) it came from.
        Website:
            https://www.kaggle.com/c/iwildcam-2020-fgvc7
        Original publication:
            @article{beery2020iwildcam,
            title={The iWildCam 2020 Competition Dataset},
            author={Beery, Sara and Cole, Elijah and Gjoka, Arvi},
            journal={arXiv preprint arXiv:2004.10340},
                    year={2020}
            }
        License:
            This dataset is distributed under Community Data License Agreement – Permissive – Version 1.0
            https://cdla.io/permissive-1-0/
        """
    _dataset_name = 'iwildcam'
    _versions_dict = {
        '2.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x6313da2b204647e79a14b468131fcd64/contents/blob/',
            'compressed_size': 11_957_420_032}}


    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load splits
        df = pd.read_csv(self._data_dir / 'metadata.csv')

        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df['split_id'].values

        # Filenames
        self._input_array = df['filename'].values

        # Labels
        self._y_array = torch.tensor(df['y'].values)
        self._n_classes = max(df['y']) + 1
        self._y_size = 1
        assert len(np.unique(df['y'])) == self._n_classes

        # Location/group info
        n_groups = max(df['location_remapped']) + 1
        self._n_groups = n_groups
        assert len(np.unique(df['location_remapped'])) == self._n_groups

        # Sequence info
        n_sequences = max(df['sequence_remapped']) + 1
        self._n_sequences = n_sequences
        assert len(np.unique(df['sequence_remapped'])) == self._n_sequences

        # Extract datetime subcomponents and include in metadata
        df['datetime_obj'] = df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        df['year'] = df['datetime_obj'].apply(lambda x: int(x.year))
        df['month'] = df['datetime_obj'].apply(lambda x: int(x.month))
        df['day'] = df['datetime_obj'].apply(lambda x: int(x.day))
        df['hour'] = df['datetime_obj'].apply(lambda x: int(x.hour))
        df['minute'] = df['datetime_obj'].apply(lambda x: int(x.minute))
        df['second'] = df['datetime_obj'].apply(lambda x: int(x.second))

        self._metadata_array = torch.tensor(np.stack([df['location_remapped'].values,
                            df['sequence_remapped'].values,
                            df['year'].values, df['month'].values, df['day'].values,
                            df['hour'].values, df['minute'].values, df['second'].values,
                            self.y_array], axis=1))
        self._metadata_fields = ['location', 'sequence', 'year', 'month', 'day', 'hour', 'minute', 'second', 'y']

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['location']))

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
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
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
                        })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
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
        img_path = self.data_dir / 'train' / self._input_array[idx]
        img = Image.open(img_path)

        return img
