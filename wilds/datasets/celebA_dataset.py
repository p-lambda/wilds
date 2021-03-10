import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class CelebADataset(WILDSDataset):
    """
    A variant of the CelebA dataset.
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to facilitate comparisons to previous work.

    Supported `split_scheme`:
        'official'

    Input (x):
        Images of celebrity faces that have already been cropped and centered.

    Label (y):
        y is binary. It is 1 if the celebrity in the image has blond hair, and is 0 otherwise.

    Metadata:
        Each image is annotated with whether the celebrity has been labeled 'Male' or 'Female'.

    Website:
        http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Original publication:
        @inproceedings{liu2015faceattributes,
          title = {Deep Learning Face Attributes in the Wild},
          author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
          booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
          month = {December},
          year = {2015}
        }

    This variant of the dataset is identical to the setup in:
        @inproceedings{sagawa2019distributionally,
          title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle = {International Conference on Learning Representations},
          year = {2019}
        }

    License:
        This version of the dataset was originally downloaded from Kaggle
        https://www.kaggle.com/jessicali9530/celeba-dataset

        It is available for non-commercial research purposes only.
    """
    _dataset_name = 'celebA'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xfe55077f5cd541f985ebf9ec50473293/contents/blob/',
            'compressed_size': 1_308_557_312}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        target_name = 'Blond_Hair'
        confounder_names = ['Male']

        # Read in attributes
        attrs_df = pd.read_csv(
            os.path.join(self.data_dir, 'list_attr_celeba.csv'))

        # Split out filenames and attribute names
        # Note: idx and filenames are off by one.
        self._input_array = attrs_df['image_id'].values
        self._original_resolution = (178, 218)
        attrs_df = attrs_df.drop(labels='image_id', axis='columns')
        attr_names = attrs_df.columns.copy()
        def attr_idx(attr_name):
            return attr_names.get_loc(attr_name)

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        attrs_df = attrs_df.values
        attrs_df[attrs_df == -1] = 0

        # Get the y values
        target_idx = attr_idx(target_name)
        self._y_array = torch.LongTensor(attrs_df[:, target_idx])
        self._y_size = 1
        self._n_classes = 2

        # Get metadata
        confounder_idx = [attr_idx(a) for a in confounder_names]
        confounders = attrs_df[:, confounder_idx]

        self._metadata_array = torch.cat(
            (torch.LongTensor(confounders), self._y_array.reshape((-1, 1))),
            dim=1)
        confounder_names = [s.lower() for s in confounder_names]
        self._metadata_fields = confounder_names + ['y']
        self._metadata_map = {
            'y': ['not blond', '    blond'] # Padding for str formatting
        }

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(confounder_names + ['y']))

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        split_df = pd.read_csv(
            os.path.join(self.data_dir, 'list_eval_partition.csv'))
        self._split_array = split_df['partition'].values

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       # Note: idx and filenames are off by one.
       img_filename = os.path.join(
           self.data_dir,
           'img_align_celeba',
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
