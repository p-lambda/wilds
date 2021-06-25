import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import DetectionAccuracy


DATASETS_DECODER = {0: 'Rres_1',
 1: 'NMBU_2',
 2: 'NMBU_1',
 3: 'Arvalis_9',
 4: 'Arvalis_11',
 5: 'Arvalis_6',
 6: 'Arvalis_5',
 7: 'Arvalis_7',
 8: 'Inrae_1',
 9: 'Arvalis_10',
 10: 'Arvalis_12',
 11: 'Arvalis_4',
 12: 'Arvalis_3',
 13: 'Arvalis_2',
 14: 'Arvalis_1',
 15: 'Arvalis_8',
 16: 'Ethz_1',
 17: 'ULiège-GxABT_1',
 18: 'Utokyo_2',
 19: 'Utokyo_1',
 20: 'Utokyo_3',
 21: 'NAU_1',
 22: 'Ukyoto_1',
 23: 'NAU_3',
 24: 'NAU_2',
 25: 'ARC_1',
 26: 'UQ_11',
 27: 'UQ_10',
 28: 'UQ_9',
 29: 'UQ_8',
 30: 'UQ_6',
 31: 'Terraref_2',
 32: 'Terraref_1',
 33: 'KSU_4',
 34: 'KSU_3',
 35: 'KSU_2',
 36: 'KSU_1',
 37: 'CIMMYT_3',
 38: 'CIMMYT_2',
 39: 'CIMMYT_1',
 40: 'UQ_6',
 41: 'UQ_5',
 42: 'UQ_4',
 43: 'UQ_3',
 44: 'UQ_2',
 45: 'UQ_1',
 46: 'Usask_1'
}

class GlobalWheatDataset(WILDSDataset):
    """
    The GlobalWheat-WILDS wheat head localization dataset.
    This is a modified version of the original Global Wheat Head Dataset 2021.

    The current version does not contain test or validation labels, as it is being used in a
    currently-running competition.
    After the competition concludes in July 2021, we will update the dataset to contain the
    final splits with test and validation labels, and add the dataset to the official WILDS
    benchmark.

    Supported `split_scheme`:
        - 'official'
    Input (x):
        1024 x 1024 RGB images of wheat field canopy starting from anthesis (flowering) to ripening.
    Output (y):
        y is a n x 4-dimensional vector where each line represents a box coordinate (x_min, y_min, x_max, y_max)
    Metadata:
        Each image is annotated with the ID of the domain (location_date_sensor) it came from (integer from 0 to 46).
    Website:
        http://www.global-wheat.com/
    Original publication:
        @article{david_global_2020,
            title = {Global {Wheat} {Head} {Detection} ({GWHD}) {Dataset}: {A} {Large} and {Diverse} {Dataset} of {High}-{Resolution} {RGB}-{Labelled} {Images} to {Develop} and {Benchmark} {Wheat} {Head} {Detection} {Methods}},
            volume = {2020},
            url = {https://doi.org/10.34133/2020/3521852},
            doi = {10.34133/2020/3521852},
            journal = {Plant Phenomics},
            author = {David, Etienne and Madec, Simon and Sadeghi-Tehran, Pouria and Aasen, Helge and Zheng, Bangyou and Liu, Shouyang and Kirchgessner, Norbert and Ishikawa, Goro and Nagasawa, Koichi and Badhon, Minhajul A. and Pozniak, Curtis and de Solan, Benoit and Hund, Andreas and Chapman, Scott C. and Baret, Frédéric and Stavness, Ian and Guo, Wei},
            month = Aug,
            year = {2020},
            note = {Publisher: AAAS},
            pages = {3521852},
        }
    License:
        This dataset is distributed under the MIT license.
    """

    _dataset_name = 'globalwheat'

    # Version 0.9 corresponds to the final dataset, but without the validation and test labels,
    # since it is being used in a currently-running competition (http://www.global-wheat.com/).
    # Users can submit their val+test predictions to the competition to obtain an estimate of
    # held-out performance computed on a fraction of those predictions;
    # please see the tutorial at https://www.aicrowd.com/challenges/global-wheat-challenge-2021.
    # We will update the dataset to include these labels and update the splits after the
    # competition ends in July 2021.
    _versions_dict = {
        '0.9': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x8ba9122a41454997afdfb78762d390cf/contents/blob/',
            'compressed_size': 10_280_247_296},
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x8ba9122a41454997afdfb78762d390cf/contents/blob/',
            'compressed_size': 10_280_247_296}
            }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):

        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (1024, 1024)
        self.root = Path(self.data_dir)
        self._is_detection = True
        self._is_classification = False
        self._y_size = None
        self._n_classes = 1

        self._split_scheme = split_scheme

        # Get filenames
        if split_scheme == "official":
            train_data_df = pd.read_csv(self.root / f'official_train.csv')
            val_data_df = pd.read_csv(self.root / f'official_val.csv')
            test_data_df = pd.read_csv(self.root / f'official_test.csv')

        elif split_scheme == "ood_with_subsampled_test":
            if version == "0.9":
                print("Warning: ood_with_subsampled_test is not available in 0.9")
            else:
                train_data_df = pd.read_csv(self.root / f'official_train.csv')
                val_data_df = pd.read_csv(self.root / f'official_val.csv')
                test_data_df = pd.read_csv(self.root / f'in_dist_test.csv')

        elif split_scheme == "in-dist":
            if version == "0.9":
                print("Warning: in-dist is not available in 0.9")
            else:
                train_data_df = pd.read_csv(self.root / f'in_dist_train.csv')
                val_data_df = pd.read_csv(self.root / f'official_val.csv')
                test_data_df = pd.read_csv(self.root / f'in_dist_test.csv')

        elif split_scheme == "fixed-train":
            if version == "0.9":
                print("Warning: fixed-train is not available in 0.9")
            else:
                train_data_df = pd.read_csv(self.root / f'fixed_train_train.csv')
                val_data_df = pd.read_csv(self.root / f'fixed_train_val.csv')
                test_data_df = pd.read_csv(self.root / f'fixed_train_test.csv')

        elif split_scheme == "fixed-test":
            if version == "0.9":
                print("Warning: fixed-test is not available in 0.9")
            else:
                train_data_df = pd.read_csv(self.root / f'fixed_test_train.csv')
                val_data_df = pd.read_csv(self.root / f'official_val.csv')
                test_data_df = pd.read_csv(self.root / f'fixed_test_test.csv')


        self._image_array = []
        self._split_array, self._y_array, self._metadata_array = [], [], []

        for i, df in enumerate([train_data_df, val_data_df, test_data_df]):
            self._image_array.extend(list(df['image_name'].values))
            boxes_string = list(df['BoxesString'].values)
            all_boxes = [GlobalWheatDataset._decode_string(box_string) for box_string in boxes_string]
            self._split_array.extend([i] * len(all_boxes))

            labels = [{
                "boxes": torch.stack([
                    torch.tensor(box)
                    for box in boxes
                ]),
                "labels": torch.tensor([1]*len(boxes)).long()
            } if len(boxes) > 0 else {
                "boxes": torch.empty(0,4),
                "labels": torch.empty(0,dtype=torch.long)
            } for boxes in all_boxes]

            self._y_array.extend(labels)
            self._metadata_array.extend([int(item) for item in df['domain'].values])

        self._split_array = np.array(self._split_array)
        self._metadata_array = torch.tensor(self._metadata_array,
                                            dtype=torch.long).unsqueeze(1)
        self._metadata_fields = ['location_date_sensor']
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['location_date_sensor'])
        self._metric = DetectionAccuracy()
        self._collate = GlobalWheatDataset._collate_fn

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       img_filename = self.root / "images" / self._image_array[idx]
       x = Image.open(img_filename)
       return x

    def eval(self, y_pred, y_true, metadata):
        """
        The main evaluation metric, detection_acc_avg_dom,
        measures the simple average of the detection accuracies
        of each domain.
        """
        results, results_str = self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

        detection_accs = []
        for k, v in results.items():
            if k.startswith('detection_acc_location_date_sensor:'):
                d = k.split(':')[1]
                count = results[f'count_location_date_sensor:{d}']
                if count > 0:
                    detection_accs.append(v)
        detection_acc_avg_dom = np.array(detection_accs).mean()
        results['detection_acc_avg_dom'] = detection_acc_avg_dom
        results_str = f'Average detection_acc across domains: {detection_acc_avg_dom:.3f}\n' + results_str
        return results, results_str

    @staticmethod
    def _decode_string(box_string):
        """
        Helper method to decode each box_string in the BoxesString field of the data CSVs
        """
        if box_string == "no_box":
            return np.zeros((0,4))
        else:
            try:
                boxes =  np.array([np.array([int(eval(i)) for i in box.split(" ")])
                            for box in box_string.split(";")])
                return boxes
            except:
                print(box_string)
                print("Submission is not well formatted. empty boxes will be returned")
                return np.zeros((0,4))

    @staticmethod
    def _collate_fn(batch):
        """
        Stack x (batch[0]) and metadata (batch[2]), but not y.
        originally, batch = (item1, item2, item3, item4)
        after zip, batch = [(item1[0], item2[0], ..), ..]
        """
        batch = list(zip(*batch))
        batch[0] = torch.stack(batch[0])
        batch[1] = list(batch[1])
        batch[2] = torch.stack(batch[2])
        return tuple(batch)
