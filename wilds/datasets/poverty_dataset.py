from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.metrics.all_metrics import MSE, PearsonCorrelation
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.utils import subsample_idxs, shuffle_arr

DATASET = '2009-17'
BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']


DHS_COUNTRIES = [
    'angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
    'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
    'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria', 'rwanda', 'senegal',
    'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe']

_SURVEY_NAMES_2009_17A = {
    'train': ['cameroon', 'democratic_republic_of_congo', 'ghana', 'kenya',
              'lesotho', 'malawi', 'mozambique', 'nigeria', 'senegal',
              'togo', 'uganda', 'zambia', 'zimbabwe'],
    'val': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
    'test': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
}
_SURVEY_NAMES_2009_17B = {
    'train': ['angola', 'cote_d_ivoire', 'democratic_republic_of_congo',
              'ethiopia', 'kenya', 'lesotho', 'mali', 'mozambique',
              'nigeria', 'rwanda', 'senegal', 'togo', 'uganda', 'zambia'],
    'val': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
    'test': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
}
_SURVEY_NAMES_2009_17C = {
    'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire', 'ethiopia',
              'guinea', 'kenya', 'lesotho', 'mali', 'rwanda', 'senegal',
              'sierra_leone', 'tanzania', 'zambia'],
    'val': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
    'test': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
}
_SURVEY_NAMES_2009_17D = {
    'train': ['angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
              'ethiopia', 'ghana', 'guinea', 'malawi', 'mali', 'rwanda',
              'sierra_leone', 'tanzania', 'zimbabwe'],
    'val': ['kenya', 'lesotho', 'senegal', 'zambia'],
    'test': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
}
_SURVEY_NAMES_2009_17E = {
    'train': ['benin', 'burkina_faso', 'cameroon', 'democratic_republic_of_congo',
              'ghana', 'guinea', 'malawi', 'mozambique', 'nigeria', 'sierra_leone',
              'tanzania', 'togo', 'uganda', 'zimbabwe'],
    'val': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
    'test': ['kenya', 'lesotho', 'senegal', 'zambia'],
}

SURVEY_NAMES = {
    '2009-17A': _SURVEY_NAMES_2009_17A,
    '2009-17B': _SURVEY_NAMES_2009_17B,
    '2009-17C': _SURVEY_NAMES_2009_17C,
    '2009-17D': _SURVEY_NAMES_2009_17D,
    '2009-17E': _SURVEY_NAMES_2009_17E,
}


# means and standard deviations calculated over the entire dataset (train + val + test),
# with negative values set to 0, and ignoring any pixel that is 0 across all bands
# all images have already been mean subtracted and normalized (x - mean) / std

_MEANS_2009_17 = {
    'BLUE':  0.059183,
    'GREEN': 0.088619,
    'RED':   0.104145,
    'SWIR1': 0.246874,
    'SWIR2': 0.168728,
    'TEMP1': 299.078023,
    'NIR':   0.253074,
    'DMSP':  4.005496,
    'VIIRS': 1.096089,
    # 'NIGHTLIGHTS': 5.101585, # nightlights overall
}

_STD_DEVS_2009_17 = {
    'BLUE':  0.022926,
    'GREEN': 0.031880,
    'RED':   0.051458,
    'SWIR1': 0.088857,
    'SWIR2': 0.083240,
    'TEMP1': 4.300303,
    'NIR':   0.058973,
    'DMSP':  23.038301,
    'VIIRS': 4.786354,
    # 'NIGHTLIGHTS': 23.342916, # nightlights overall
}


def split_by_countries(idxs, ood_countries, metadata):
    countries = np.asarray(metadata['country'].iloc[idxs])
    is_ood = np.any([(countries == country) for country in ood_countries], axis=0)
    return idxs[~is_ood], idxs[is_ood]


class PovertyMapDataset(WILDSDataset):
    """
    The PovertyMap poverty measure prediction dataset.
    This is a processed version of LandSat 5/7/8 satellite imagery originally from Google Earth Engine under the names `LANDSAT/LC08/C01/T1_SR`,`LANDSAT/LE07/C01/T1_SR`,`LANDSAT/LT05/C01/T1_SR`,
    nighttime light imagery from the DMSP and VIIRS satellites (Google Earth Engine names `NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4` and `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`)
    and processed DHS survey metadata obtained from https://github.com/sustainlab-group/africa_poverty and originally from `https://dhsprogram.com/data/available-datasets.cfm`.

    Supported `split_scheme`:
        - 'official' and `countries`, which are equivalent
        - 'mixed-to-test'

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
    _dataset_name = 'poverty'
    _versions_dict = {
        '1.1': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xfc0aa86ad9af4eb08c42dfc40eacf094/contents/blob/',
            'compressed_size': 13_091_823_616}}

    def __init__(self, version=None, root_dir='data', download=False,
                 split_scheme='official',
                 no_nl=False, fold='A',
                 use_ood_val=True,
                 cache_size=100):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (224, 224)

        self._split_dict = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
        self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}

        if split_scheme == 'official':
            split_scheme = 'countries'

        if split_scheme == 'mixed-to-test':
            self.oracle_training_set = True
        elif split_scheme in ['official', 'countries']:
            self.oracle_training_set = False
        else:
            raise ValueError("Split scheme not recognized")
        self._split_scheme = split_scheme

        self.no_nl = no_nl
        if fold not in {'A', 'B', 'C', 'D', 'E'}:
            raise ValueError("Fold must be A, B, C, D, or E")

        self.root = Path(self._data_dir)
        self.metadata = pd.read_csv(self.root / 'dhs_metadata.csv')
        # country folds, split off OOD
        country_folds = SURVEY_NAMES[f'2009-17{fold}']

        self._split_array = -1 * np.ones(len(self.metadata))

        incountry_folds_split = np.arange(len(self.metadata))
        # take the test countries to be ood
        idxs_id, idxs_ood_test = split_by_countries(incountry_folds_split, country_folds['test'], self.metadata)
        # also create a validation OOD set
        idxs_id, idxs_ood_val = split_by_countries(idxs_id, country_folds['val'], self.metadata)
        for split in ['test', 'val', 'id_test', 'id_val', 'train']:
            # keep ood for test, otherwise throw away ood data
            if split == 'test':
                idxs = idxs_ood_test
            elif split == 'val':
                idxs = idxs_ood_val
            else:
                idxs = idxs_id
                num_eval = 2000
                # if oracle, sample from all countries
                if split == 'train' and self.oracle_training_set:
                    idxs = subsample_idxs(incountry_folds_split, num=len(idxs_id), seed=ord(fold))[num_eval:]
                elif split == 'train':
                    idxs = subsample_idxs(idxs, take_rest=True, num=num_eval, seed=ord(fold))
                else:
                    eval_idxs  = subsample_idxs(idxs, take_rest=False, num=num_eval, seed=ord(fold))

                if split != 'train':
                    if split == 'id_val':
                        idxs = eval_idxs[:num_eval//2]
                    else:
                        idxs = eval_idxs[num_eval//2:]
            self._split_array[idxs] = self._split_dict[split]

        if not use_ood_val:
            self._split_dict = {'train': 0, 'val': 1, 'id_test': 2, 'ood_val': 3, 'test': 4}
            self._split_names = {'train': 'Train', 'val': 'ID Val', 'id_test': 'ID Test', 'ood_val': 'OOD Val', 'test': 'OOD Test'}

        self._y_array = torch.from_numpy(np.asarray(self.metadata['wealthpooled'])[:, np.newaxis]).float()
        self._y_size = 1

        # add country group field
        country_to_idx = {country: i for i, country in enumerate(DHS_COUNTRIES)}
        self.metadata['country'] = [country_to_idx[country] for country in self.metadata['country'].tolist()]
        self._metadata_map = {'country': DHS_COUNTRIES}
        self._metadata_array = torch.from_numpy(self.metadata[['urban', 'wealthpooled', 'country']].astype(float).to_numpy())
        # rename wealthpooled to y
        self._metadata_fields = ['urban', 'y', 'country']

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['urban'])

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

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model
            - y_true (LongTensor): Ground-truth values
            - metadata (Tensor): Metadata
            - prediction_fn (function): Only None supported
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        assert prediction_fn is None, "PovertyMapDataset.eval() does not support prediction_fn"

        metrics = [MSE(), PearsonCorrelation()]

        all_results = {}
        all_results_str = ''
        for metric in metrics:
            results, results_str = self.standard_group_eval(
                metric,
                self._eval_grouper,
                y_pred, y_true, metadata)
            all_results.update(results)
            all_results_str += results_str
        return all_results, all_results_str
