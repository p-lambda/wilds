import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class WaterbirdsDataset(WILDSDataset):
    """
    The Waterbirds dataset.
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to facilitate comparisons to previous work.

    Supported `split_scheme`:
        'official'

    Input (x):
        Images of birds against various backgrounds that have already been cropped and centered.

    Label (y):
        y is binary. It is 1 if the bird is a waterbird (e.g., duck), and 0 if it is a landbird.

    Metadata:
        Each image is annotated with whether the background is a land or water background.

    Original publication:
        @inproceedings{sagawa2019distributionally,
          title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle = {International Conference on Learning Representations},
          year = {2019}
        }

    The dataset was constructed from the CUB-200-2011 dataset and the Places dataset:
        @techreport{WahCUB_200_2011,
        	Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
        	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
        	Year = {2011}
        	Institution = {California Institute of Technology},
        	Number = {CNS-TR-2011-001}
        }
        @article{zhou2017places,
          title = {Places: A 10 million Image Database for Scene Recognition},
          author = {Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
          journal ={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          year = {2017},
          publisher = {IEEE}
        }

    License:
        The use of this dataset is restricted to non-commercial research and educational purposes.
    """

    _dataset_name = 'waterbirds'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/',
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        # Note: metadata_df is one-indexed.
        metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self._y_array = torch.LongTensor(metadata_df['y'].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata_df['place'].values), self._y_array),
            dim=1
        )
        self._metadata_fields = ['background', 'y']
        self._metadata_map = {
            'background': [' land', 'water'], # Padding for str formatting
            'y': [' landbird', 'waterbird']
        }

        # Extract filenames
        self._input_array = metadata_df['img_filename'].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = metadata_df['split'].values

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['background', 'y']))

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

        results, results_str = self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

        # For Waterbirds, the validation and test sets are constructed to be more balanced
        # compared to the training set.
        # To compute the actual average accuracy over the empirical (training) distribution,
        # we therefore weight each groups according to their frequency in the training set.

        results['adj_acc_avg'] = (
            (results['acc_y:landbird_background:land'] * 3498
            + results['acc_y:landbird_background:water'] * 184
            + results['acc_y:waterbird_background:land'] * 56
            + results['acc_y:waterbird_background:water'] * 1057) /
            (3498 + 184 + 56 + 1057))

        del results['acc_avg']
        results_str = f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n" + '\n'.join(results_str.split('\n')[1:])

        return results, results_str
