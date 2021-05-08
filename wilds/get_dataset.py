import wilds

def get_dataset(dataset, version=None, **dataset_kwargs):
    """
    Returns the appropriate WILDS dataset class.
    Input:
        dataset (str): Name of the dataset
        version (str): Dataset version number, e.g., '1.0'.
                       Defaults to the latest version.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified WILDSDataset class.
    """
    if version is not None:
        version = str(version)

    if dataset not in wilds.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {wilds.supported_datasets}.')

    if dataset == 'amazon':
        from wilds.datasets.amazon_dataset import AmazonDataset
        return AmazonDataset(version=version, **dataset_kwargs)

    elif dataset == 'camelyon17':
        from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
        return Camelyon17Dataset(version=version, **dataset_kwargs)

    elif dataset == 'celebA':
        from wilds.datasets.celebA_dataset import CelebADataset
        return CelebADataset(version=version, **dataset_kwargs)

    elif dataset == 'civilcomments':
        from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
        return CivilCommentsDataset(version=version, **dataset_kwargs)

    elif dataset == 'iwildcam':
        if version == '1.0':
            from wilds.datasets.archive.iwildcam_v1_0_dataset import IWildCamDataset
        else:
            from wilds.datasets.iwildcam_dataset import IWildCamDataset
        return IWildCamDataset(version=version, **dataset_kwargs)

    elif dataset == 'waterbirds':
        from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
        return WaterbirdsDataset(version=version, **dataset_kwargs)

    elif dataset == 'yelp':
        from wilds.datasets.yelp_dataset import YelpDataset
        return YelpDataset(version=version, **dataset_kwargs)

    elif dataset == 'ogb-molpcba':
        from wilds.datasets.ogbmolpcba_dataset import OGBPCBADataset
        return OGBPCBADataset(version=version, **dataset_kwargs)

    elif dataset == 'poverty':
        if version == '1.0':
            from wilds.datasets.archive.poverty_v1_0_dataset import PovertyMapDataset
        else:            
            from wilds.datasets.poverty_dataset import PovertyMapDataset
        return PovertyMapDataset(version=version, **dataset_kwargs)

    elif dataset == 'fmow':
        if version == '1.0':
            from wilds.datasets.archive.fmow_v1_0_dataset import FMoWDataset
        else:
            from wilds.datasets.fmow_dataset import FMoWDataset
        return FMoWDataset(version=version, **dataset_kwargs)

    elif dataset == 'bdd100k':
        from wilds.datasets.bdd100k_dataset import BDD100KDataset
        return BDD100KDataset(version=version, **dataset_kwargs)

    elif dataset == 'py150':
        from wilds.datasets.py150_dataset import Py150Dataset
        return Py150Dataset(version=version, **dataset_kwargs)

    elif dataset == 'sqf':
        from wilds.datasets.sqf_dataset import SQFDataset
        return SQFDataset(version=version, **dataset_kwargs)
