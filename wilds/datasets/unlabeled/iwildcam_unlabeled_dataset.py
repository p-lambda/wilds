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

        # Load splits
        df = pd.read_csv(self._data_dir / 'metadata.csv')

        # Splits
        self._split_dict = {'extra_unlabeled': 0}
        self._split_names = {'extra_unlabeled': 'Extra Unlabeled'}
        df['split_id'] = 0
        self._split_array = df['split_id'].values

        # Filenames
        df['filename'] = df['uid'].apply(lambda x: x + '.jpg')
        self._input_array = df['filename'].values

        # Location/group info
        n_groups = df['location_remapped'].nunique()
        self._n_groups = n_groups
        self._y_array = df['category_id']
        self._y_size = 1

        def get_date(x):
            if isinstance(x, str):
                return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
            else:
                return -1

        ## Extract datetime subcomponents and include in metadata
        df['datetime_obj'] = df['datetime'].apply(get_date)
        df['year'] = df['datetime_obj'].apply(lambda x: int(x.year) if isinstance(x, datetime) else -1)
        df['month'] = df['datetime_obj'].apply(lambda x: int(x.month) if isinstance(x, datetime) else -1)
        df['day'] = df['datetime_obj'].apply(lambda x: int(x.day) if isinstance(x, datetime) else -1)
        df['hour'] = df['datetime_obj'].apply(lambda x: int(x.hour) if isinstance(x, datetime) else -1)
        df['minute'] = df['datetime_obj'].apply(lambda x: int(x.minute) if isinstance(x, datetime) else -1)
        df['second'] = df['datetime_obj'].apply(lambda x: int(x.second) if isinstance(x, datetime) else -1)

        df['y'] = df['y'] # same mapping as in iwildcam_v2.0. -1 means the category was not in iwildcam_v2.0

        self._metadata_array = torch.tensor(np.stack([df['location_remapped'].values,
                            df['sequence_remapped'].values,
                            df['year'].values, df['month'].values, df['day'].values,
                            df['hour'].values, df['minute'].values, df['second'].values,
                            df['y']], axis=1))
        self._metadata_fields = ['location', 'sequence', 'year', 'month', 'day', 'hour', 'minute', 'second', 'y']
        self._metadata_map = {}

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['location']))

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / 'images' / self._input_array[idx]
        img = Image.open(img_path)

        return img
