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
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
from wilds.datasets.fmow_dataset import categories

Image.MAX_IMAGE_PIXELS = 10000000000

class FMoWUnlabeledDataset(WILDSUnlabeledDataset):
    """
    The FMoW-WILDS land use / building classification dataset.
    This is a processed version of the Functional Map of the World dataset originally sourced from https://github.com/fMoW/dataset.

    Support `split_scheme`
        'official': official split, which is equivalent to 'time_after_2016'
        `time_after_{YEAR}` for YEAR between 2002--2018

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
    _dataset_name = 'fmow_unlabeled'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/',
            'compressed_size': 53_893_324_800,
            "equivalent_dataset": "fmow_v1.1",}
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', seed=111, use_ood_val=True):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        if split_scheme=='official':
            split_scheme='time_after_2016'
        self._split_scheme = split_scheme

        self.root = Path(self._data_dir)
        self.seed = int(seed)
        self._original_resolution = (224, 224)

        self.metadata = pd.read_csv(self.root / 'rgb_metadata.csv')
        country_codes_df = pd.read_csv(self.root / 'country_code_mapping.csv')
        countrycode_to_region = {k: v for k, v in zip(country_codes_df['alpha-3'], country_codes_df['region'])}
        regions = [countrycode_to_region.get(code, 'Other') for code in self.metadata['country_code'].to_list()]
        self.metadata['region'] = regions
        all_countries = self.metadata['country_code']

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

        if self.split_scheme.startswith('time_after'):
            self._split_dict = {
                    "train_unlabeled": 10,
                    "val_unlabeled": 11,
                    "test_unlabeled": 12,
            }
            self._split_names = {
                "train_unlabeled": "Unlabeled Train",
                "val_unlabeled": "Unlabeled Validation",
                "test_unlabeled": "Unlabeled Test",
            }
        else:
            raise ValueError(f"Split scheme {self.split_scheme} is not recognized.")

        test_mask = np.asarray(self.metadata['split'] == 'test')
        val_mask = np.asarray(self.metadata['split'] == 'val')
        seq_mask = np.asarray(self.metadata['split'] == 'seq')
        self._split_array = -1 * np.ones(len(self.metadata))
        for split in self._split_dict.keys():
            # unused data from labeled FMoW
            if split == 'test_unlabeled':
                test_unlabeled_mask = self.test_ood_mask & ~test_mask & ~val_mask
                idxs = np.arange(len(self.metadata))[test_unlabeled_mask]

            elif split == 'val_unlabeled':
                val_unlabeled_mask = self.val_ood_mask & ~test_mask & ~val_mask
                idxs = np.arange(len(self.metadata))[val_unlabeled_mask]

            elif split == 'train_unlabeled':
                train_unlabeled_mask = seq_mask & ~self.ood_mask
                idxs = np.arange(len(self.metadata))[train_unlabeled_mask]

            self._split_array[idxs] = self._split_dict[split]
        unlabeled_mask = (self._split_array != -1)
        self.full_idxs = np.arange(len(self.metadata))[unlabeled_mask]
        self._split_array = self._split_array[unlabeled_mask]

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

        # hidden labels
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)} 
        self.metadata['y'] = np.asarray([self.category_to_idx[y] for y in list(self.metadata['category'])])
        self._y_array = torch.LongTensor(self.metadata['y'].values)[unlabeled_mask]

        self._metadata_fields = ['region', 'year', 'y']
        self._metadata_array = torch.from_numpy(self.metadata[self._metadata_fields].astype(int).to_numpy()).long()[unlabeled_mask]

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        idx = self.full_idxs[idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img
