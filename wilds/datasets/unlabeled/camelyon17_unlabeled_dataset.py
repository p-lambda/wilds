import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wilds.datasets.camelyon17_dataset import TEST_CENTER, VAL_CENTER
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
from wilds.common.grouper import CombinatorialGrouper


class Camelyon17UnlabeledDataset(WILDSUnlabeledDataset):
    """
    Unlabeled Camelyon17-WILDS dataset.
    This dataset contains patches from all of the slides in the original CAMELYON17 training data,
    except for the slides that were labeled with lesion annotations and therefore used in the
    labeled Camelyon17Dataset.

    Supported `split_scheme`:
        'official'

    Input (x):
        96x96 image patches extracted from histopathology slides.

    Metadata:
        Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
        and the slide it came from (integer from 0 to 49).

    Website:
        https://camelyon17.grand-challenge.org/

    Original publication:
        @article{bandi2018detection,
          title={From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge},
          author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
          journal={IEEE transactions on medical imaging},
          volume={38},
          number={2},
          pages={550--560},
          year={2018},
          publisher={IEEE}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    _dataset_name = "camelyon17_unlabeled"
    _versions_dict = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0xa78be8a88a00487a92006936514967d2/contents/blob/",
            "compressed_size": 69_442_379_933,
        }
    }

    def __init__(
        self, version=None, root_dir="data", download=False, split_scheme="official"
    ):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (96, 96)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, "metadata.csv"),
            index_col=0,
            dtype={"patient": "str"},
        )

        # Get filenames
        self._input_array = [
            f"patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
            for patient, node, x, y in self._metadata_df.loc[
                :, ["patient", "node", "x_coord", "y_coord"]
            ].itertuples(index=False, name=None)
        ]

        self._split_scheme = split_scheme
        if self._split_scheme == "official":
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
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")

        # Extract splits
        centers = self._metadata_df["center"].values.astype("long")
        num_centers = int(np.max(centers)) + 1
        self._metadata_df["split"] = self.split_dict["train_unlabeled"]
        val_center_mask = self._metadata_df["center"] == VAL_CENTER
        test_center_mask = self._metadata_df["center"] == TEST_CENTER
        self._metadata_df.loc[val_center_mask, "split"] = self.split_dict[
            "val_unlabeled"
        ]
        self._metadata_df.loc[test_center_mask, "split"] = self.split_dict[
            "test_unlabeled"
        ]
        # Centers 1 and 2 have 600,030 unlabeled examples each.
        # The rest of the unlabeled data is used for the train_unlabeled split (1,799,247 total).
        assert self._metadata_df.loc[val_center_mask].shape[0] == 600_030
        assert self._metadata_df.loc[test_center_mask].shape[0] == 600_030
        train_center_mask = ~self._metadata_df["center"].isin([VAL_CENTER, TEST_CENTER])
        assert self._metadata_df.loc[train_center_mask].shape[0] == 1_799_247

        self._split_array = self._metadata_df["split"].values

        self._y_array = 100 * torch.LongTensor(self._metadata_df["tumor"].values) # in metadata.csv, these are all -1
        self._metadata_array = torch.stack(
            (
                torch.LongTensor(centers),
                torch.LongTensor(self._metadata_df["slide"].values),
                self._y_array,
            ),
            dim=1,
        )
        self._metadata_fields = ["hospital", "slide", "y"]

        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=["slide"]
        )

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = os.path.join(self.data_dir, self._input_array[idx])
        x = Image.open(img_filename).convert("RGB")
        return x
