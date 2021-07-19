import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class Camelyon17Dataset(WILDSDataset):
    """
    The CAMELYON17-WILDS histopathology dataset.
    This is a modified version of the original CAMELYON17 dataset.

    Supported `split_scheme`:
        - 'official'
        - 'mixed-to-test'

    Input (x):
        96x96 image patches extracted from histopathology slides.

    Label (y):
        y is binary. It is 1 if the central 32x32 region contains any tumor tissue, and 0 otherwise.

    Metadata:
        Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
        and the slide it came from (integer from 0 to 49).

    Website:
        https://camelyon17.grand-challenge.org/

    Original publication:
        @article{bandi2018detection,
          title={From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge},
          author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
          journal={IEEE transactions on medical imaging},
          volume={38},
          number={2},
          pages={550--560},
          year={2018},
          publisher={IEEE}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    _dataset_name = 'camelyon17'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/',
            'compressed_size': 10_658_709_504}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (96,96)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['tumor'].values)
        self._y_size = 1
        self._n_classes = 2

        # Get filenames
        self._input_array = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            self._metadata_df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]

        # Extract splits
        # Note that the hospital numbering here is different from what's in the paper,
        # where to avoid confusing readers we used a 1-indexed scheme and just labeled the test hospital as 5.
        # Here, the numbers are 0-indexed.
        test_center = 2
        val_center = 1

        self._split_dict = {
            'train': 0,
            'id_val': 1,
            'test': 2,
            'val': 3
        }
        self._split_names = {
            'train': 'Train',
            'id_val': 'Validation (ID)',
            'test': 'Test',
            'val': 'Validation (OOD)',
        }
        centers = self._metadata_df['center'].values.astype('long')
        num_centers = int(np.max(centers)) + 1
        val_center_mask = (self._metadata_df['center'] == val_center)
        test_center_mask = (self._metadata_df['center'] == test_center)
        self._metadata_df.loc[val_center_mask, 'split'] = self.split_dict['val']
        self._metadata_df.loc[test_center_mask, 'split'] = self.split_dict['test']

        self._split_scheme = split_scheme
        if self._split_scheme == 'official':
            pass
        elif self._split_scheme == 'mixed-to-test':
            # For the mixed-to-test setting,
            # we move slide 23 (corresponding to patient 042, node 3 in the original dataset)
            # from the test set to the training set
            slide_mask = (self._metadata_df['slide'] == 23)
            self._metadata_df.loc[slide_mask, 'split'] = self.split_dict['train']
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = self._metadata_df['split'].values

        self._metadata_array = torch.stack(
            (torch.LongTensor(centers),
             torch.LongTensor(self._metadata_df['slide'].values),
             self._y_array),
            dim=1)
        self._metadata_fields = ['hospital', 'slide', 'y']

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['slide'])

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       img_filename = os.path.join(
           self.data_dir,
           self._input_array[idx])
       x = Image.open(img_filename).convert('RGB')
       return x

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
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
