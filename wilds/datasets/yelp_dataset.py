import os, csv
import torch
import pandas as pd
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.utils import map_to_id_array
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper

NOT_IN_DATASET = -1

class YelpDataset(WILDSDataset):
    """
    Yelp dataset.
    This is a modified version of the Yelp Open Dataset
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to reproduce observations discussed in the WILDS paper.

    Supported `split_scheme`:
        'official': official split, which is equivalent to 'time'
        'time': shifts from reviews written before 2013 to reviews written after 2013
        'user': shifts to unseen reviewers
        'time_baseline': oracle baseline splits for time shifts

    Input (x):
        Review text of maximum token length of 512.

    Label (y):
        y is the star rating (0,1,2,3,4 corresponding to 1-5 stars)

    Metadata:
        user: reviewer ID
        year: year in which the review was written
        business: business ID
        city: city of the business
        state: state of the business

    Website:
        https://www.yelp.com/dataset

    License:
        Because of the Dataset License provided by Yelp, we are unable to redistribute the data.
        Please download the data through the website (https://www.yelp.com/dataset/download) by
        agreeing to the Dataset License.
    """
    _dataset_name = 'yelp'
    _versions_dict = {
        '1.0': {
            'download_url': None,
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        # set variables        
        self._version = version
        if split_scheme=='official':
            split_scheme = 'time'
        self._split_scheme = split_scheme
        self._y_type = 'long'
        self._y_size = 1
        self._n_classes = 5
        # path
        self._data_dir = self.initialize_data_dir(root_dir, download)
        # Load data
        data_df = pd.read_csv(os.path.join(self.data_dir, 'reviews.csv'),
                dtype={'review_id': str, 'user_id':str, 'business_id':str, 'stars':int, 'useful':int, 'funny':int,
                       'cool':int, 'text':str, 'date':str, 'year':int, 'city':str, 'state':str, 'categories':str},
                keep_default_na=False, na_values=[], quoting=csv.QUOTE_NONNUMERIC)
        split_df = pd.read_csv(os.path.join(self.data_dir, 'splits', f'{self.split_scheme}.csv'))
        is_in_dataset = split_df['split']!=NOT_IN_DATASET
        split_df = split_df[is_in_dataset]
        data_df = data_df[is_in_dataset]
        # Get arrays
        self._split_array = split_df['split'].values
        self._input_array = list(data_df['text'])
        # Get metadata
        self._metadata_fields, self._metadata_array, self._metadata_map = self.load_metadata(data_df, self.split_array)
        # Get y from metadata
        self._y_array = getattr(self.metadata_array[:,self.metadata_fields.index('y')], self._y_type)()
        # Set split info
        self.initialize_split_dicts()
        # eval
        self.initialize_eval_grouper()
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._input_array[idx]

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
        if self.split_scheme=='user':
            # first compute groupwise accuracies
            g = self._eval_grouper.metadata_to_group(metadata)
            results = {
                **metric.compute(y_pred, y_true),
                **metric.compute_group_wise(y_pred, y_true, g, self._eval_grouper.n_groups)
            }
            accs = []
            for group_idx in range(self._eval_grouper.n_groups):
                group_str = self._eval_grouper.group_field_str(group_idx)
                group_metric = results.pop(metric.group_metric_field(group_idx))
                group_counts = results.pop(metric.group_count_field(group_idx))
                results[f'{metric.name}_{group_str}'] = group_metric
                results[f'count_{group_str}'] = group_counts
                if group_counts>0:
                    accs.append(group_metric)
            accs = np.array(accs)
            results['10th_percentile_acc'] = np.percentile(accs, 10)
            results[f'{metric.worst_group_metric_field}'] = metric.worst(accs)
            results_str = (
                f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
                f"10th percentile {metric.name}: {results['10th_percentile_acc']:.3f}\n"
                f"Worst-group {metric.name}: {results[metric.worst_group_metric_field]:.3f}\n"
            )
            return results, results_str
        else:
            return self.standard_group_eval(
                metric,
                self._eval_grouper,
                y_pred, y_true, metadata)

    def initialize_split_dicts(self):
        if self.split_scheme in ('user', 'time'):
            self._split_dict = {'train': 0, 'val': 1, 'id_val': 2, 'test': 3, 'id_test': 4}
            self._split_names = {'train': 'Train', 'val': 'Validation (OOD)', 'id_val': 'Validation (ID)', 'test':'Test (OOD)', 'id_test': 'Test (ID)'}
        elif self.split_scheme in ('time_baseline',):
            # use defaults
            pass
        else:
            raise ValueError(f'Split scheme {self.split_scheme} not recognized')

    def load_metadata(self, data_df, split_array):
        # Get metadata
        columns = ['user_id', 'business_id', 'year', 'city', 'state', 'stars',]
        metadata_fields = ['user', 'business', 'year', 'city', 'state', 'y']
        metadata_df = data_df[columns].copy()
        metadata_df.columns = metadata_fields
        sort_idx = np.argsort(split_array)
        ordered_maps = {}
        for field in ['user', 'business', 'city', 'state']:
            # map to IDs in the order of split values
            ordered_maps[field] = pd.unique(metadata_df.iloc[sort_idx][field])
        ordered_maps['y'] = range(1,6)
        ordered_maps['year'] = range(metadata_df['year'].min(), metadata_df['year'].max()+1)
        metadata_map, metadata = map_to_id_array(metadata_df, ordered_maps)
        return metadata_fields, torch.from_numpy(metadata.astype('long')), metadata_map

    def initialize_eval_grouper(self):
        if self.split_scheme=='user':
            self._eval_grouper = CombinatorialGrouper(
                dataset=self,
                groupby_fields=['user'])
        elif self.split_scheme in ('time', 'time_baseline'):
            self._eval_grouper = CombinatorialGrouper(
                dataset=self,
                groupby_fields=['year'])
        else:
            raise ValueError(f'Split scheme {self.split_scheme} not recognized')
