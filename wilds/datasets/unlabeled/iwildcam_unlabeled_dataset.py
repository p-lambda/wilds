from datetime import datetime
from pathlib import Path
import os

from PIL import Image
import pandas as pd
import numpy as np
import torch
import json

from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1


class IWildCamUnlabeledDataset(WILDSUnlabeledDataset):
    """
        The iWildCam2020 dataset.
        This is a modified version of the original iWildCam2020 competition dataset.
        Input (x):
            RGB images from camera traps
        Metadata:
            Each image is annotated with the ID of the location (camera trap) it came from.
        Website:
            http://lila.science/datasets/wcscameratraps
            https://library.wcs.org/ScienceData/Camera-Trap-Data-Summary.aspx
        Original publication:
            Wildlife Conservation Society Camera Traps
            Dataset. http://lila.science/datasets/
            wcscameratraps
        License:
            This dataset is distributed under Community Data License Agreement – Permissive – Version 1.0
            https://cdla.io/permissive-1-0/
        """
    _dataset_name = 'iwildcam_unlabeled'
    _versions_dict = {
        '1.0': {
            'download_url': 'TODO',
            'compressed_size': 0,
            'equivalent_dataset': 'iwildcam_unlabeled_v1.0'}}


    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        print("data dir: ", self._data_dir)

        # Load splits
        df = pd.read_csv(self._data_dir / 'metadata.csv')

        # Splits
        self._split_dict = {'extra_unlabeled': 0}
        self._split_names = {'extra_unlabeled': 'Extra Unlabeled'}

        # df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        df['split_id'] = 0
        self._split_array = df['split_id'].values

        # Filenames
        self._input_array = df['file_name'].values

        def get_loc_as_int(loc):
            if loc == 'unknown':
                return 889

            try:
                return int(loc)
            except Exception as e:
                print("e: ", e)
                return 888

        # Location/group info
        df['location'] = df['location'].apply(get_loc_as_int)
        df['y_array'] = 888
        n_groups = df['location'].nunique()
        self._n_groups = n_groups
        self._y_array = df['y_array']
        self._y_size = 1
        #assert len(np.unique(df['location'])) == self._n_groups

        ## Sequence info
        df['sequence'] = 1
        #n_sequences = max(df['sequence_remapped']) + 1
        self._n_sequences = 1
        #assert len(np.unique(df['sequence_remapped'])) == self._n_sequences

        ## Extract datetime subcomponents and include in metadata
        #df['datetime_obj'] = df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        #df['year'] = df['datetime_obj'].apply(lambda x: int(x.year))
        #df['month'] = df['datetime_obj'].apply(lambda x: int(x.month))
        #df['day'] = df['datetime_obj'].apply(lambda x: int(x.day))
        #df['hour'] = df['datetime_obj'].apply(lambda x: int(x.hour))
        #df['minute'] = df['datetime_obj'].apply(lambda x: int(x.minute))
        #df['second'] = df['datetime_obj'].apply(lambda x: int(x.second))

        df['datetime_obj'] = -1
        df['year'] = -1
        df['month'] = -1
        df['day'] = -1
        df['hour'] = -1
        df['minute'] = -1
        df['second'] = -1

        df['y'] = -1


        print(" df unique: ", df['location'].unique())

        self._metadata_array = torch.tensor(np.stack([df['location'].values], axis=1))
        self._metadata_fields = ['location']
        self._metadata_map = {}


        self._metadata_array = torch.tensor(np.stack([df['location'].values,
                            df['sequence'].values,
                            df['year'].values, df['month'].values, df['day'].values,
                            df['hour'].values, df['minute'].values, df['second'].values,
                            df['y']], axis=1))
        self._metadata_fields = ['location', 'sequence', 'year', 'month', 'day', 'hour', 'minute', 'second', 'y']

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['location']))

        super().__init__(root_dir, download, split_scheme)

    # def eval(self, y_pred, y_true, metadata, prediction_fn=None):
    #     """
    #     Computes all evaluation metrics.
    #     Args:
    #         - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
    #                            But they can also be other model outputs such that prediction_fn(y_pred)
    #                            are predicted labels.
    #         - y_true (LongTensor): Ground-truth labels
    #         - metadata (Tensor): Metadata
    #         - prediction_fn (function): A function that turns y_pred into predicted labels
    #     Output:
    #         - results (dictionary): Dictionary of evaluation metrics
    #         - results_str (str): String summarizing the evaluation metrics
    #     """
    #     metrics = [
    #         Accuracy(prediction_fn=prediction_fn),
    #         Recall(prediction_fn=prediction_fn, average='macro'),
    #         F1(prediction_fn=prediction_fn, average='macro'),
    #     ]

    #     results = {}

    #     for i in range(len(metrics)):
    #         results.update({
    #             **metrics[i].compute(y_pred, y_true),
    #                     })

    #     results_str = (
    #         f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
    #         f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
    #         f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
    #     )

    #     return results, results_str

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
