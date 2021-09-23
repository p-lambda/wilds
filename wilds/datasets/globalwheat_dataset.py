import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import DetectionAccuracy

SESSIONS = [
    'Arvalis_1',
    'Arvalis_2',
    'Arvalis_3',
    'Arvalis_4',
    'Arvalis_5',
    'Arvalis_6',
    'Arvalis_7',
    'Arvalis_8',
    'Arvalis_9',
    'Arvalis_10',
    'Arvalis_11',
    'Arvalis_12',
    'ETHZ_1',
    'Inrae_1',
    'NMBU_1',
    'NMBU_2',
    'Rres_1',
    'ULiège-GxABT_1',
    'Utokyo_1',
    'Utokyo_2',
    'Utokyo_3',
    'Ukyoto_1',
    'NAU_1',
    'NAU_2',
    'NAU_3',
    'ARC_1',
    'UQ_1',
    'UQ_2',
    'UQ_3',
    'UQ_4',
    'UQ_5',
    'UQ_6',
    'UQ_7',
    'UQ_8',
    'UQ_9',
    'UQ_10',
    'UQ_11',
    'Terraref_1',
    'Terraref_2',
    'KSU_1',
    'KSU_2',
    'KSU_3',
    'KSU_4',
    'CIMMYT_1',
    'CIMMYT_2',
    'CIMMYT_3',
    'Usask_1'
]

COUNTRIES = [
    'Switzerland',
    'UK',
    'Belgium',
    'Norway',
    'France',
    'Canada',
    'US',
    'Mexico',
    'Japan',
    'China',
    'Australia',
    'Sudan',
]

LOCATIONS = [
    'Baima',
    'Brookstead',
    'Ciudad Obregon',
    'Gatton',
    'Gembloux',
    'Gréoux',
    'KSU',
    'Kyoto',
    'Maricopa, AZ',
    'McAllister',
    'Mons',
    'NARO-Hokkaido',
    'NARO-Tsukuba',
    'NMBU',
    'Rothamsted',
    'Saskatchewan',
    'Toulouse',
    'Usask',
    'VLB',
    'VSC',
    'Wad Medani',
]

STAGES = [
    'Filling',
    'Filling - Ripening',
    'multiple',
    'Post-flowering',
    'Post-Flowering',
    'Ripening',
]

class GlobalWheatDataset(WILDSDataset):
    """
    The GlobalWheat-WILDS wheat head localization dataset.
    This is a modified version of the original Global Wheat Head Dataset 2021.

    Supported `split_scheme`:
        - 'official'
        - 'official_with_subsampled_test'
        - 'test-to-test'
        - 'mixed-to-test'
    Input (x):
        1024 x 1024 RGB images of wheat field canopy starting from anthesis (flowering) to ripening.
    Output (y):
        y is a n x 4-dimensional vector where each line represents a box coordinate (x_min, y_min, x_max, y_max)
    Metadata:
        Each image is annotated with the ID of the domain (session) it came from (integer from 0 to 46).
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
        @misc{david2021global,
            title={Global Wheat Head Dataset 2021: more diversity to improve the benchmarking of wheat head localization methods},
            author={Etienne David and Mario Serouart and Daniel Smith and Simon Madec and Kaaviya Velumani and Shouyang Liu and Xu Wang and Francisco Pinto Espinosa and Shahameh Shafiee and Izzat S. A. Tahir and Hisashi Tsujimoto and Shuhei Nasuda and Bangyou Zheng and Norbert Kichgessner and Helge Aasen and Andreas Hund and Pouria Sadhegi-Tehran and Koichi Nagasawa and Goro Ishikawa and Sébastien Dandrifosse and Alexis Carlier and Benoit Mercatoris and Ken Kuroki and Haozhou Wang and Masanori Ishii and Minhajul A. Badhon and Curtis Pozniak and David Shaner LeBauer and Morten Lilimo and Jesse Poland and Scott Chapman and Benoit de Solan and Frédéric Baret and Ian Stavness and Wei Guo},
            year={2021},
            eprint={2105.07660},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }
    License:
        This dataset is distributed under the MIT license.
    """

    _dataset_name = 'globalwheat'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x443fbcb18eeb4f80b5ea4a9f77795168/contents/blob/',
            'compressed_size': 10_286_120_960}
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

        self._split_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
        }
        self._split_names = {
            'train': 'Train',
            'val': 'Validation (OOD)',
            'test':'Test (OOD)',
        }

        data_dfs = {}
        if split_scheme == "official":
            data_dfs['train'] = pd.read_csv(self.root / f'official_train.csv')
            data_dfs['val'] = pd.read_csv(self.root / f'official_val.csv')
            data_dfs['test'] = pd.read_csv(self.root / f'official_test.csv')
            data_dfs['id_val'] = pd.read_csv(self.root / f'fixed_train_val.csv')
            data_dfs['id_test'] = pd.read_csv(self.root / f'fixed_train_test.csv')
            self._split_dict = {
                'train': 0,
                'val': 1,
                'test': 2,
                'id_val': 3,
                'id_test': 4,
            }
            self._split_names = {
                'train': 'Train',
                'val': 'Validation (OOD)',
                'test':'Test (OOD)',
                'id_val': 'Validation (ID)',
                'id_test': 'Test (ID)'
            }

        elif split_scheme == "official_with_subsampled_test":
            data_dfs['train'] = pd.read_csv(self.root / f'official_train.csv')
            data_dfs['val'] = pd.read_csv(self.root / f'official_val.csv')
            data_dfs['test'] = pd.read_csv(self.root / f'fixed_test_test.csv')

        elif split_scheme == "test-to-test":
            data_dfs['train'] = pd.read_csv(self.root / f'fixed_test_train.csv')
            data_dfs['val'] = pd.read_csv(self.root / f'official_val.csv')
            data_dfs['test'] = pd.read_csv(self.root / f'fixed_test_test.csv')

        elif split_scheme == "mixed-to-test":
            data_dfs['train'] = pd.read_csv(self.root / f'mixed_train_train.csv')
            data_dfs['val'] = pd.read_csv(self.root / f'official_val.csv')
            data_dfs['test'] = pd.read_csv(self.root / f'mixed_train_test.csv')

        else:
            raise ValueError(f'Split scheme {self.split_scheme} not recognized')

        self._image_array = []
        self._split_array, self._y_array, self._metadata_array = [], [], []

        for split_name, split_idx in self._split_dict.items():
            df = data_dfs[split_name]
            self._image_array.extend(list(df['image_name'].values))
            boxes_string = list(df['BoxesString'].values)
            all_boxes = [GlobalWheatDataset._decode_string(box_string) for box_string in boxes_string]
            self._split_array.extend([split_idx] * len(all_boxes))

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
        self._metadata_array = torch.cat(
            (self._metadata_array,
            torch.zeros(
                (len(self._metadata_array), 3),
                dtype=torch.long)),
            dim=1)

        domain_df = pd.read_csv(self.root / 'metadata_domain.csv', sep=';')
        for session_idx, session_name in enumerate(SESSIONS):
            idx = pd.Index(domain_df['name']).get_loc(session_name)
            country = domain_df.loc[idx, 'country']
            location = domain_df.loc[idx, 'location']
            stage = domain_df.loc[idx, 'development_stage']

            session_mask = (self._metadata_array[:, 0] == session_idx)
            self._metadata_array[session_mask, 1] = COUNTRIES.index(country)
            self._metadata_array[session_mask, 2] = LOCATIONS.index(location)
            self._metadata_array[session_mask, 3] = STAGES.index(stage)

        self._metadata_fields = ['session', 'country', 'location', 'stage']
        self._metadata_map = {
            'session': SESSIONS,
            'country': COUNTRIES,
            'location': LOCATIONS,
            'stage': STAGES,
        }

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['session'])
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
            if k.startswith('detection_acc_session:'):
                d = k.split(':')[1]
                count = results[f'count_session:{d}']
                if count > 0:
                    detection_accs.append(v)
        detection_acc_avg_dom = np.array(detection_accs).mean()
        results['detection_acc_avg_dom'] = detection_acc_avg_dom
        results_str = f'Average detection_acc across session: {detection_acc_avg_dom:.3f}\n' + results_str
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
