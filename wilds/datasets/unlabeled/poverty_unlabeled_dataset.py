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

Image.MAX_IMAGE_PIXELS = 10000000000

from wilds.datasets.poverty_dataset import (
        DATASET,
        BAND_ORDER,
        DHS_COUNTRIES,
        SURVEY_NAMES,
        _MEANS_2009_17,
        _STD_DEVS_2009_17,
        split_by_countries
        )


class PovertyMapUnlabeledDataset(WILDSUnlabeledDataset):
    """
    The unlabeled PovertyMap-WILDS poverty measure prediction dataset.
    This is a processed version of LandSat 5/7/8 satellite imagery originally from Google Earth Engine under the names `LANDSAT/LC08/C01/T1_SR`,`LANDSAT/LE07/C01/T1_SR`,`LANDSAT/LT05/C01/T1_SR`,
    nighttime light imagery from the DMSP and VIIRS satellites (Google Earth Engine names `NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4` and `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`)
    and processed DHS survey metadata obtained from https://github.com/sustainlab-group/africa_poverty and originally from `https://dhsprogram.com/data/available-datasets.cfm`.
    Unlabeled data are sampled from around DHS survey locations.

    Supported `split_scheme`:
        'official' and `countries`, which are equivalent

    Input (x):
        224 x 224 x 8 satellite image, with 7 channels from LandSat and 1 nighttime light channel from DMSP/VIIRS. Already mean/std normalized.

    Output (y):
        y is a real-valued asset wealth index. Higher index corresponds to more asset wealth.

    Metadata:
        each image is annotated with location coordinates (noised for anonymity), survey year, urban/rural classification, country, nighttime light mean, nighttime light median.

    Website: https://github.com/sustainlab-group/africa_poverty

    Original publication:
    @article{yeh2020using,
        author = {Yeh, Christopher and Perez, Anthony and Driscoll, Anne and Azzari, George and Tang, Zhongyi and Lobell, David and Ermon, Stefano and Burke, Marshall},
        day = {22},
        doi = {10.1038/s41467-020-16185-w},
        issn = {2041-1723},
        journal = {Nature Communications},
        month = {5},
        number = {1},
        title = {{Using publicly available satellite imagery and deep learning to understand economic well-being in Africa}},
        url = {https://www.nature.com/articles/s41467-020-16185-w},
        volume = {11},
        year = {2020}
    }

    License:
        LandSat/DMSP/VIIRS data is U.S. Public Domain.

    """
    _dataset_name = 'poverty_unlabeled'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xdfcf71b4f6164cc1a7edb0cbb7444c8c/contents/blob/',
            'compressed_size': 172_742_430_134,
        }
    }

    def __init__(self, version=None, root_dir='data', download=False,
                 split_scheme='official',
                 no_nl=False, fold='A',
                 use_ood_val=True,
                 cache_size=100):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (224, 224)

        if split_scheme=='official':
            split_scheme = 'countries'

        self._split_scheme = split_scheme
        if self._split_scheme == 'countries':
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
            raise ValueError("Split scheme not recognized")

        self.no_nl = no_nl
        if fold not in {'A', 'B', 'C', 'D', 'E'}:
            raise ValueError("Fold must be A, B, C, D, or E")

        self.root = Path(self._data_dir)
        self.metadata = pd.read_csv(self.root / 'unlabeled_metadata.csv')
        country_folds = SURVEY_NAMES[f'2009-17{fold}']

        self._split_array = -1 * np.ones(len(self.metadata))

        incountry_folds_split = np.arange(len(self.metadata))
        # take the test countries to be ood
        idxs_id, idxs_ood_test = split_by_countries(incountry_folds_split, country_folds['test'], self.metadata)
        # also create a validation OOD set
        idxs_id, idxs_ood_val = split_by_countries(idxs_id, country_folds['val'], self.metadata)

        self._split_array[idxs_id] = self._split_dict['train_unlabeled']
        self._split_array[idxs_ood_val] = self._split_dict['val_unlabeled']
        self._split_array[idxs_ood_test] = self._split_dict['test_unlabeled']

        # no labels
        self.metadata['y'] = (-100 * np.ones(len(self.metadata)))
        # no urban/rural classification
        self.metadata['urban'] = (-100 * np.ones(len(self.metadata)))

        # add country group field
        country_to_idx = {country: i for i, country in enumerate(DHS_COUNTRIES)}
        self.metadata['country'] = [country_to_idx[country] for country in self.metadata['country'].tolist()]
        self._metadata_map = {'country': DHS_COUNTRIES}
        # rename wealthpooled to y
        self._metadata_fields = ['urban', 'y', 'country']
        self._metadata_array = torch.from_numpy(self.metadata[self._metadata_fields].astype(float).to_numpy())
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img = np.load(self.root / 'images' / f'landsat_poverty_img_{idx}.npz')['x']
        if self.no_nl:
            img[-1] = 0
        img = torch.from_numpy(img).float()

        return img
