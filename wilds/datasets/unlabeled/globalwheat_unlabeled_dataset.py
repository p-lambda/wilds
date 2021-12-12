import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.globalwheat_dataset import GlobalWheatDataset

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
    'Usask_1',
    "unlabeled_CIMMYT_PHY_9SAT_HEAT_RGB_20200415",
    "unlabeled_CIMMYT_PHY_9SAT_HEAT_RGB_20200422",
    "unlabeled_CIMMYT_PHY_BestPT_HEAT_RGB_20200422",
    "unlabeled_CIMMYT_PHY_BestPT_HEAT_RGB_280420",
    "unlabeled_CIMMYT_PHY_BestPT_RGO_RGB_20200319",
    "unlabeled_CIMMYT_PHY_HIBAPIII_HEAT_RGB_20200422",
    "unlabeled_CIMMYT_PHY_HIBAPIII_RGO_RGB_20200311",
    "unlabeled_CIMMYT_PHY_HIBAPIII_RGO_RGB_20200312",
    "unlabeled_CIMMYT_PHY_HIBAPIII_RGO_RGB_20200313",
    "unlabeled_CIMMYT_PHY_HIBAPIII_RGO_RGB_20200503",
    "unlabeled_CIMMYT_PHY_ROOTS-ANAT_HEAT_RGB_20200422",
    "unlabeled_CIMMYT_PHY_ROOTS_ANAT_RGO_RGB_20200313",
    "unlabeled_CIMMYT_PHY_ROOTS_ANAT_SQ_RGB_20200313",
    "unlabeled_KSU_17ASH_LakinFuller_KEY_PCTHEAD_20170423",
    "unlabeled_KSU__16ASH_AM-PANEL_KEY_PCTHEAD_20160504",
    "unlabeled_RRes-Dataset1",
    "unlabeled_RRes-Dataset2",
    "unlabeled_RRes-Dataset3",
    "unlabeled_RRes-Dataset4",
    "unlabeled_arvalis_greoux_Session_2020-05-25_12-13-52",
    "unlabeled_arvalis_greoux_Session_2020-06-22_07-46-39",
    "unlabeled_arvalis_VLB_Session 2020-06-02 09-33-05_VBProb",
    "unlabeled_arvalis_BIGNAN_Session 2021-06-17 13-49-26",
    "unlabeled_arvalis_ENCRAMBADE_Session 2021-05-18 07-59-37",
    "unlabeled_arvalis_ENCRAMBADE_Session 2021-06-11 07-48-21",
    "unlabeled_arvalis_OLM_Session 2021-06-02 07-47-38",
    "unlabeled_arvalis_OLM_Session 2021-06-17 11-55-36",
    "unlabeled_arvalis_VLB_Session 2021-05-28 07-08-58",
    "unlabeled_arvalis_VLB_Session 2021-06-14 08-51-33",
    "unlabeled_arvalis_mons_2018",
    "unlabeled_hokkaido_SK_SK_2021_7_20",
    "unlabeled_hokkaido_SK_SK_2021_7_28",
    "unlabeled_hokkaido_SM_SM_2021_7_10",
    "unlabeled_hokkaido_Tsujino_2021_6_16",
    "unlabeled_hokkaido_Tsujino_2021_6_23",
    "unlabeled_hokkaido_Tsujino_2021_6_7",
    "unlabeled_hokkaido_Tsujino_2021_6_9",
    "unlabeled_hokkaido_Tsujino_2021_7_10",
    "unlabeled_hokkaido_Tsujino_2021_7_11",
    "unlabeled_hokkaido_Tsujino_2021_7_20",
    "unlabeled_hokkaido_Tsujino_2021_7_3",
    "unlabeled_inrae_clermont",
    "unlabeled_uliege_6_11",
    "unlabeled_uliege_6_15",
    "unlabeled_uliege_6_16",
    "unlabeled_uliege_6_18",
    "unlabeled_uliege_6_23",
    "unlabeled_uliege_6_26",
    "unlabeled_uliege_7_13",
    "unlabeled_uliege_7_7",
    "unlabeled_usask_2019_08_06_sampled",
    "unlabeled_usask_2019_08_12_sampled",
    "unlabeled_ETHZ_2"
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
    'Eschikon',
    'Bignan',
    'Clermont',
    'Encrambade',
    'NMBU',
    'OLM',    
]

STAGES = [
    'Filling',
    'Filling-Ripening',
    'multiple',
    'Post-flowering',
    'Ripening',
    'Emergence',
]

class GlobalWheatUnlabeledDataset(WILDSUnlabeledDataset):
    """
    The GlobalWheat-WILDS wheat head localization dataset.
    This is a modified version of the original Global Wheat Head Dataset 2021.

    Supported `split_scheme`:
        - 'official'
        - 'official_with_subsampled_test'
        - 'fixed-test'
        - 'mixed-train'
    Input (x):
        1024 x 1024 RGB images of wheat field canopy starting from anthesis (flowering) to ripening.
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

    _dataset_name = "globalwheat_unlabeled"
    _versions_dict = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x59d4c1b8b7ad45cc83e080d11d1eaf94/contents/blob/",
            "compressed_size": 103_766_940_000,
        }
    }

    def __init__(
        self, version=None, root_dir="data", download=False, split_scheme="official"
    ):

        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (1024, 1024)
        self.root = Path(self.data_dir)

        self._n_classes = 1
        self._split_scheme = split_scheme

        data_dfs = {}

        if self._split_scheme == "official":
            self._split_dict = {
                "train_unlabeled": 10,
                "val_unlabeled": 11,
                "test_unlabeled": 12,
                "extra_unlabeled": 13,
            }
            self._split_names = {
                "train_unlabeled": "Unlabeled Train",
                "val_unlabeled": "Unlabeled Validation",
                "test_unlabeled": "Unlabeled Test",
                "extra_unlabeled": "Unlabeled Extra",
            }

            data_dfs["train_unlabeled"] = pd.read_csv(
                self.root / f"official_train_unlabeled.csv"
            )

            data_dfs["val_unlabeled"] = pd.read_csv(
                self.root / f"official_val_unlabeled.csv"
            )

            data_dfs["test_unlabeled"] = pd.read_csv(
                self.root / f"official_test_unlabeled.csv"
            )

            data_dfs["extra_unlabeled"] = pd.read_csv(
                self.root / f"official_extra_unlabeled.csv"
            )

        else:
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")

        self._image_array = []
        self._split_array = []
        self._metadata_array = []

        # Extract splits

        for split_name, split_idx in self._split_dict.items():
            df = data_dfs[split_name]

            self._image_array.extend(list(df["image_name"].values))
            self._split_array.extend([split_idx] * len(df))

            self._metadata_array.extend([int(item) for item in df["domain"].values])

        self._split_array = np.array(self._split_array)
        self._metadata_array = torch.tensor(
            self._metadata_array, dtype=torch.long
        ).unsqueeze(1)
        self._metadata_array = torch.cat(
            (
                self._metadata_array,
                torch.zeros((len(self._metadata_array), 3), dtype=torch.long),
            ),
            dim=1,
        )

        domain_df = pd.read_csv(self.root / "metadata_domain_unlabeled.csv", sep=";")

        for session_idx, session_name in enumerate(SESSIONS):
            idx = pd.Index(domain_df["name"]).get_loc(session_name)

            country = domain_df.loc[idx, "country"]
            location = domain_df.loc[idx, "location"]
            stage = domain_df.loc[idx, "development_stage"]

            session_mask = self._metadata_array[:, 0] == session_idx

            self._metadata_array[session_mask, 1] = COUNTRIES.index(country)
            self._metadata_array[session_mask, 2] = LOCATIONS.index(location)
            self._metadata_array[session_mask, 3] = STAGES.index(stage)

        self._metadata_fields = ["session", "country", "location", "stage"]
        self._metadata_map = {
            "session": SESSIONS,
            "country": COUNTRIES,
            "location": LOCATIONS,
            "stage": STAGES,
        }

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = self.root / "images" / self._image_array[idx]
        x = Image.open(img_filename)
        return x
