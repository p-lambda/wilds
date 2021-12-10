import os

import torch
import numpy as np

from wilds.datasets.wilds_dataset import WILDSDataset


class WILDSUnlabeledDataset(WILDSDataset):
    """
    Shared dataset class for all unlabeled WILDS datasets.
    Each data point in the dataset is an (x, metadata) tuple, where:
    - x is the input features
    - metadata is a vector of relevant information, e.g., domain.
    """

    # The corresponding indices for the unlabeled splits should not overlap with
    # the indices of their labeled counterparts (indices start from 0).
    # So, for unlabeled splits, the indices should start from 10.
    DEFAULT_SPLITS = {
        "train_unlabeled": 10,
        "val_unlabeled": 11,
        "test_unlabeled": 12,
        "extra_unlabeled": 13,
    }
    DEFAULT_SPLIT_NAMES = {
        "train_unlabeled": "Unlabeled Train",
        "val_unlabeled": "Unlabeled Validation",
        "test_unlabeled": "Unlabeled Test",
        "extra_unlabeled": "Unlabeled Extra",
    }
    DEFAULT_SOURCE_DOMAIN_SPLITS = [10]

    _UNSUPPORTED_FUNCTIONALITY_ERROR = "Not supported - no labels available."

    def __len__(self):
        return len(self.metadata_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        metadata = self.metadata_array[idx]
        return x, metadata

    def get_subset(self, split, frac=1.0, transform=None, load_y=False):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return WILDSUnlabeledSubset(self, split_idx, transform, load_y=load_y)

    def check_init(self):
        """
        Convenience function to check that the WILDSDataset is properly configured.
        """
        required_attrs = [
            "_dataset_name",
            "_data_dir",
            "_split_scheme",
            "_split_array",
            "_metadata_fields",
            "_metadata_array",
        ]
        for attr_name in required_attrs:
            assert hasattr(
                self, attr_name
            ), f"WILDSUnlabeledDataset is missing {attr_name}."

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        # Check splits
        assert self.split_dict.keys() == self.split_names.keys()

        # Check that required arrays are Tensors
        assert isinstance(
            self.metadata_array, torch.Tensor
        ), "metadata_array must be a torch.Tensor"

        # Check that dimensions match
        assert len(self.split_array) == len(self.metadata_array)

        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]

    def initialize_data_dir(self, root_dir, download):
        if "equivalent_dataset" in self.versions_dict[self.version]:
            self.check_version()
            os.makedirs(root_dir, exist_ok=True)

            # If the dataset has an equivalent dataset, check if the equivalent dataset already exists
            # at the root directory. If it does, don't download and just return the equivalent dataset path.
            data_dir = os.path.join(
                root_dir, self.versions_dict[self.version]["equivalent_dataset"]
            )
            if not os.path.exists(data_dir):
                # Proceed with downloading the equivalent dataset.
                self.download_dataset(data_dir, download)
            return data_dir
        else:
            return super().initialize_data_dir(root_dir, download)

    def eval(self, y_pred, y_true, metadata):
        raise AttributeError(WILDSUnlabeledDataset._UNSUPPORTED_FUNCTIONALITY_ERROR)

    @property
    def y_array(self):
        raise AttributeError(WILDSUnlabeledDataset._UNSUPPORTED_FUNCTIONALITY_ERROR)

    @property
    def y_size(self):
        raise AttributeError(WILDSUnlabeledDataset._UNSUPPORTED_FUNCTIONALITY_ERROR)

    @property
    def split_dict(self):
        """
        A dictionary mapping splits to integer identifiers (used in split_array),
        Keys should match up with split_names.
        """
        return getattr(self, "_split_dict", WILDSUnlabeledDataset.DEFAULT_SPLITS)

    @property
    def split_names(self):
        """
        A dictionary mapping splits to their pretty names,
        Keys should match up with split_dict.
        """
        return getattr(self, "_split_names", WILDSUnlabeledDataset.DEFAULT_SPLIT_NAMES)

    @property
    def source_domain_splits(self):
        """
        List of split IDs that are from the source domain.
        """
        return getattr(
            self,
            "_source_domain_splits",
            WILDSUnlabeledDataset.DEFAULT_SOURCE_DOMAIN_SPLITS,
        )


class WILDSUnlabeledSubset(WILDSUnlabeledDataset):
    def __init__(self, dataset, indices, transform, load_y=False):
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = [
            "_dataset_name",
            "_data_dir",
            "_collate",
            "_split_scheme",
            "_split_dict",
            "_split_names",
            "_metadata_fields",
            "_metadata_map",
            "_y_array", 
        ]
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.load_y = load_y

    def __getitem__(self, idx):
        x, metadata = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        if self.load_y:
            y = self._y_array[self.indices[idx]]
            return x, y, metadata
        else:
            return x, metadata

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

class WILDSPseudolabeledSubset(WILDSUnlabeledDataset):
    """Pseudolabeled subset initialized from an unlabeled subset"""
    def __init__(self, reference_subset, pseudolabels, transform, collate=None):
        assert len(reference_subset) == len(pseudolabels)
        self.pseudolabels = pseudolabels
        copied_attrs = [
            "dataset",
            "indices",
            "_dataset_name",
            "_data_dir",
            "_collate",
            "_split_scheme",
            "_split_dict",
            "_split_names",
            "_metadata_fields",
            "_metadata_map",
        ]
        for attr_name in copied_attrs:
            if hasattr(reference_subset, attr_name):
                setattr(self, attr_name, getattr(reference_subset, attr_name, None))
        self.transform = transform
        if collate:
            self._collate = collate

    def __getitem__(self, idx):
        x, metadata = self.dataset[self.indices[idx]]
        y_pseudo = self.pseudolabels[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y_pseudo, metadata

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]