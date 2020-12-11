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
    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        # set variables
        self._dataset_name = 'yelp'
        self._version = '1.0'
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
        self._metric = Accuracy()
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._input_array[idx]

    def eval(self, y_pred, y_true, metadata):
        # first compute groupwise accuracies
        g = self._eval_grouper.metadata_to_group(metadata)
        results = {
            **self._metric.compute(y_pred, y_true),
            **self._metric.compute_group_wise(y_pred, y_true, g, self._eval_grouper.n_groups)
        }
        # then do specific computations for each split and make pretty string
        if self.split_scheme in ('time', 'time_baseline'):
            results_str = (
                f"Average {self._metric.name}: {results[self._metric.agg_metric_field]:.3f}\n"
            )
            for group_idx in range(self._eval_grouper.n_groups):
                if results[self._metric.group_count_field(group_idx)]==0:
                    continue
                results_str += (
                        f'  {self._eval_grouper.group_str(group_idx)} [n = {results[self._metric.group_count_field(group_idx)]:6.0f}]:\t {self._metric.name} = {results[self._metric.group_metric_field(group_idx)]:5.3f}\n'
                )
        elif self.split_scheme=='user':
            accs = []
            for group_idx in range(self._eval_grouper.n_groups):
                if results[self._metric.group_count_field(group_idx)]>0:
                    accs.append(results[self._metric.group_metric_field(group_idx)])
            accs = np.array(accs)
            results['10th_percentile_acc'] = np.percentile(accs, 10)
            results_str = (
                f"Average {self._metric.name}: {results[self._metric.agg_metric_field]:.3f}\n"
                f"10th percentile {self._metric.name}: {results['10th_percentile_acc']:.3f}\n"
            )
        else:
            raise ValueError(f'Split scheme {self.split_scheme} not recognized')
        return results, results_str

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
