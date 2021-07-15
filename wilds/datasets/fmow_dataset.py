from pathlib import Path
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
import tarfile
import datetime
import pytz
from PIL import Image
from tqdm import tqdm
from wilds.common.utils import subsample_idxs
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

Image.MAX_IMAGE_PIXELS = 10000000000


categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]


class FMoWDataset(WILDSDataset):
    """
    The Functional Map of the World land use / building classification dataset.
    This is a processed version of the Functional Map of the World dataset originally sourced from https://github.com/fMoW/dataset.

    Supported `split_scheme`:
        - 'official': official split, which is equivalent to 'time_after_2016'
        - 'mixed-to-test'
        - 'time_after_{YEAR}' for YEAR between 2002--2018

    Input (x):
        224 x 224 x 3 RGB satellite image.

    Label (y):
        y is one of 62 land use / building classes

    Metadata:
        each image is annotated with a location coordinate, timestamp, country code. This dataset computes region as a derivative of country code.

    Website: https://github.com/fMoW/dataset

    Original publication:
    @inproceedings{fmow2018,
      title={Functional Map of the World},
      author={Christie, Gordon and Fendley, Neil and Wilson, James and Mukherjee, Ryan},
      booktitle={CVPR},
      year={2018}
    }

    License:
        Distributed under the FMoW Challenge Public License.
        https://github.com/fMoW/dataset/blob/master/LICENSE

    """
    _dataset_name = 'fmow'
    _versions_dict = {
        '1.1': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/',
            'compressed_size': 53_893_324_800}
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', seed=111, use_ood_val=True):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        self._split_dict = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
        self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}

        self.oracle_training_set = False
        if split_scheme == 'official':
            split_scheme = 'time_after_2016'
        elif split_scheme == 'mixed-to-test':
            split_scheme = 'time_after_2016'
            self.oracle_training_set = True
        self._split_scheme = split_scheme

        self.root = Path(self._data_dir)
        self.seed = int(seed)
        self._original_resolution = (224, 224)

        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}

        self.metadata = pd.read_csv(self.root / 'rgb_metadata.csv')
        country_codes_df = pd.read_csv(self.root / 'country_code_mapping.csv')
        countrycode_to_region = {k: v for k, v in zip(country_codes_df['alpha-3'], country_codes_df['region'])}
        regions = [countrycode_to_region.get(code, 'Other') for code in self.metadata['country_code'].to_list()]
        self.metadata['region'] = regions
        all_countries = self.metadata['country_code']

        self.num_chunks = 101
        self.chunk_size = len(self.metadata) // (self.num_chunks - 1)

        if self._split_scheme.startswith('time_after'):
            year = int(self._split_scheme.split('_')[2])
            year_dt = datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)
            self.test_ood_mask = np.asarray(pd.to_datetime(self.metadata['timestamp']) >= year_dt)
            # use 3 years of the training set as validation
            year_minus_3_dt = datetime.datetime(year-3, 1, 1, tzinfo=pytz.UTC)
            self.val_ood_mask = np.asarray(pd.to_datetime(self.metadata['timestamp']) >= year_minus_3_dt) & ~self.test_ood_mask
            self.ood_mask = self.test_ood_mask | self.val_ood_mask
        else:
            raise ValueError(f"Not supported: self._split_scheme = {self._split_scheme}")

        self._split_array = -1 * np.ones(len(self.metadata))
        for split in self._split_dict.keys():
            idxs = np.arange(len(self.metadata))
            if split == 'test':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                idxs = idxs[self.test_ood_mask & test_mask]
            elif split == 'val':
                val_mask = np.asarray(self.metadata['split'] == 'val')
                idxs = idxs[self.val_ood_mask & val_mask]
            elif split == 'id_test':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                idxs = idxs[~self.ood_mask & test_mask]
            elif split == 'id_val':
                val_mask = np.asarray(self.metadata['split'] == 'val')
                idxs = idxs[~self.ood_mask & val_mask]
            else:
                split_mask = np.asarray(self.metadata['split'] == split)
                idxs = idxs[~self.ood_mask & split_mask]

            if self.oracle_training_set and split == 'train':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                unused_ood_idxs = np.arange(len(self.metadata))[self.ood_mask & ~test_mask]
                subsample_unused_ood_idxs = subsample_idxs(unused_ood_idxs, num=len(idxs)//2, seed=self.seed+2)
                subsample_train_idxs = subsample_idxs(idxs.copy(), num=len(idxs) // 2, seed=self.seed+3)
                idxs = np.concatenate([subsample_unused_ood_idxs, subsample_train_idxs])
            self._split_array[idxs] = self._split_dict[split]

        if not use_ood_val:
            self._split_dict = {'train': 0, 'val': 1, 'id_test': 2, 'ood_val': 3, 'test': 4}
            self._split_names = {'train': 'Train', 'val': 'ID Val', 'id_test': 'ID Test', 'ood_val': 'OOD Val', 'test': 'OOD Test'}

        # filter out sequestered images from full dataset
        seq_mask = np.asarray(self.metadata['split'] == 'seq')
        # take out the sequestered images
        self._split_array = self._split_array[~seq_mask]
        self.full_idxs = np.arange(len(self.metadata))[~seq_mask]

        self._y_array = np.asarray([self.category_to_idx[y] for y in list(self.metadata['category'])])
        self.metadata['y'] = self._y_array
        self._y_array = torch.from_numpy(self._y_array).long()[~seq_mask]
        self._y_size = 1
        self._n_classes = 62

        # convert region to idxs
        all_regions = list(self.metadata['region'].unique())
        region_to_region_idx = {region: i for i, region in enumerate(all_regions)}
        self._metadata_map = {'region': all_regions}
        region_idxs = [region_to_region_idx[region] for region in self.metadata['region'].tolist()]
        self.metadata['region'] = region_idxs

        # make a year column in metadata
        year_array = -1 * np.ones(len(self.metadata))
        ts = pd.to_datetime(self.metadata['timestamp'])
        for year in range(2002, 2018):
            year_mask = np.asarray(ts >= datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)) \
                        & np.asarray(ts < datetime.datetime(year+1, 1, 1, tzinfo=pytz.UTC))
            year_array[year_mask] = year - 2002
        self.metadata['year'] = year_array
        self._metadata_map['year'] = list(range(2002, 2018))

        self._metadata_fields = ['region', 'year', 'y']
        self._metadata_array = torch.from_numpy(self.metadata[self._metadata_fields].astype(int).to_numpy()).long()[~seq_mask]

        self._eval_groupers = {
            'year': CombinatorialGrouper(dataset=self, groupby_fields=['year']),
            'region': CombinatorialGrouper(dataset=self, groupby_fields=['region']),
        }

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        idx = self.full_idxs[idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

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
        # Overall evaluation + evaluate by year
        all_results, all_results_str = self.standard_group_eval(
            metric,
            self._eval_groupers['year'],
            y_pred, y_true, metadata)
        # Evaluate by region and ignore the "Other" region
        region_grouper = self._eval_groupers['region']
        region_results = metric.compute_group_wise(
            y_pred,
            y_true,
            region_grouper.metadata_to_group(metadata),
            region_grouper.n_groups)
        all_results[f'{metric.name}_worst_year'] = all_results.pop(metric.worst_group_metric_field)
        region_metric_list = []
        for group_idx in range(region_grouper.n_groups):
            group_str = region_grouper.group_field_str(group_idx)
            group_metric = region_results[metric.group_metric_field(group_idx)]
            group_counts = region_results[metric.group_count_field(group_idx)]
            all_results[f'{metric.name}_{group_str}'] = group_metric
            all_results[f'count_{group_str}'] = group_counts
            if region_results[metric.group_count_field(group_idx)] == 0 or "Other" in group_str:
                continue
            all_results_str += (
                f'  {region_grouper.group_str(group_idx)}  '
                f"[n = {region_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {region_results[metric.group_metric_field(group_idx)]:5.3f}\n")
            region_metric_list.append(region_results[metric.group_metric_field(group_idx)])
        all_results[f'{metric.name}_worst_region'] = metric.worst(region_metric_list)
        all_results_str += f"Worst-group {metric.name}: {all_results[f'{metric.name}_worst_region']:.3f}\n"

        return all_results, all_results_str
