from wilds.datasets.py150_dataset import Py150Dataset

class Py150AugDataset(Py150Dataset):
    """
    The Py150AugDataset is a subclass of Py150Dataset that uses augmented data.
    The augmented data is expected to be in the 'data-aug' directory instead of 'data'.
    """

    def __init__(self, version=None, root_dir='data-aug', download=False, split_scheme='official'):
        # Initialize the parent class with the modified root directory
        super().__init__(version=version, root_dir=root_dir, download=download, split_scheme=split_scheme)

    # If there are any other specific changes needed for the augmented dataset,
    # such as different handling of metadata, different evaluation metrics, etc.,
    # those methods would be overridden here.
