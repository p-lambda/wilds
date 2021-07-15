import os
from pathlib import Path
from collections import defaultdict

from PIL import Image
import pandas as pd
import numpy as np
import torch

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy


class RxRx1Dataset(WILDSDataset):
    """
    The RxRx1-WILDS dataset.
    This is a modified version of the original RxRx1 dataset.

    Supported `split_scheme`:
        - 'official'
        - 'mixed-to-test'

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
        @inproceedings{taylor2019rxrx1,
            author = {Taylor, J. and Earnshaw, B. and Mabey, B. and Victors, M. and  Yosinski, J.},
            title = {RxRx1: An Image Set for Cellular Morphological Variation Across Many Experimental Batches.},
            year = {2019},
            booktitle = {International Conference on Learning Representations (ICLR)},
            booksubtitle = {AI for Social Good Workshop},
            url = {https://aiforsocialgood.github.io/iclr2019/accepted/track1/pdfs/30_aisg_iclr2019.pdf},
        }

    License:
        This work is licensed under a Creative Commons
        Attribution-NonCommercial-ShareAlike 4.0 International License. To view
        a copy of this license, visit
        http://creativecommons.org/licenses/by-nc-sa/4.0/.
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
        if self._split_scheme not in ['official', 'mixed-to-test']:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load splits
        df = pd.read_csv(self._data_dir / 'metadata.csv')

        # Splits
        if split_scheme == 'official':
            # Training:   33 experiments, 1 site per experiment (site 1)
            # Validation: 4 experiments, 2 sites per experiment
            # Test OOD:   14 experiments, 2 sites per experiment
            # Test ID:    Same 33 experiments from training set
            #             1 site per experiment (site 2)
            self._split_dict = {
                'train': 0,
                'val': 1,
                'test': 2,
                'id_test': 3
            }
            self._split_names = {
                'train': 'Train',
                'val': 'Validation (OOD)',
                'test': 'Test (OOD)',
                'id_test': 'Test (ID)'
            }
            self._split_array = df.dataset.apply(self._split_dict.get).values
            # id_test set
            mask = ((df.dataset == 'train') & (df.site == 2)).values
            self._split_array[mask] = self.split_dict['id_test']

        elif split_scheme == 'mixed-to-test':
            # Training:   33 experiments total, 1 site per experiment (site 1)
            #             = 19 experiments from the orig training set (site 1)
            #             + 14 experiments from the orig test set (site 1)
            # Validation: same as official split
            # Test:       14 experiments from the orig test set,
            #             1 site per experiment (site 2)
            self._split_dict = {
                'train': 0,
                'val': 1,
                'test': 2
            }
            self._split_names = {
                'train': 'Train',
                'val': 'Validation',
                'test': 'Test'
            }
            self._split_array = df.dataset.apply(self._split_dict.get).values
            # Use half of the training set (site 1) and discard site 2
            mask_to_discard = ((df.dataset == 'train') & (df.site == 2)).values
            self._split_array[mask_to_discard] = -1
            # Take all site 1 in the test set and move it to train
            mask_to_move = ((df.dataset == 'test') & (df.site == 1)).values
            self._split_array[mask_to_move] = self._split_dict['train']
            # For each of the test experiments, remove a train experiment of the same cell type
            test_cell_type_counts = defaultdict(int)
            test_experiments = df.loc[(df['dataset'] == 'test'), 'experiment'].unique()
            for test_experiment in test_experiments:
                test_cell_type = test_experiment.split('-')[0]
                test_cell_type_counts[test_cell_type] += 1
            # Training experiments are numbered starting from 1 and left-padded with 0s
            experiments_to_discard = [
                f'{cell_type}-{num:02}'
                for cell_type, count in test_cell_type_counts.items()
                for num in range(1, count+1)]
            # Sanity check
            train_experiments = df.loc[(df['dataset'] == 'train'), 'experiment'].unique()
            for experiment in experiments_to_discard:
                assert experiment in train_experiments
                mask_to_discard = (df.experiment == experiment).values
                self._split_array[mask_to_discard] = -1
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

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
