from typing import Optional

import wilds

def get_dataset(dataset: str, version: Optional[str] = None, unlabeled: bool = False, **dataset_kwargs):
    """
    Returns the appropriate WILDS dataset class.
    Input:
        dataset (str): Name of the dataset
        version (Union[str, None]): Dataset version number, e.g., '1.0'.
                                    Defaults to the latest version.
        unlabeled (bool): If true, use the unlabeled version of the dataset.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified WILDSDataset class.
    """
    if version is not None:
        version = str(version)

    if dataset not in wilds.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {wilds.supported_datasets}.')

    if unlabeled and dataset not in wilds.unlabeled_datasets:
        raise ValueError(f'Unlabeled data is not available for {dataset}. Must be one of {wilds.unlabeled_datasets}.')

    if dataset == 'amazon':
        if unlabeled:
            from wilds.datasets.unlabeled.amazon_unlabeled_dataset import AmazonUnlabeledDataset
            return AmazonUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            from wilds.datasets.amazon_dataset import AmazonDataset
            return AmazonDataset(version=version, **dataset_kwargs)

    elif dataset == 'camelyon17':
        if unlabeled:
            from wilds.datasets.unlabeled.camelyon17_unlabeled_dataset import Camelyon17UnlabeledDataset
            return Camelyon17UnlabeledDataset(version=version, **dataset_kwargs)
        else:
            from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
            return Camelyon17Dataset(version=version, **dataset_kwargs)

    elif dataset == 'celebA':
        from wilds.datasets.celebA_dataset import CelebADataset
        return CelebADataset(version=version, **dataset_kwargs)

    elif dataset == 'civilcomments':
        if unlabeled:
            from wilds.datasets.unlabeled.civilcomments_unlabeled_dataset import CivilCommentsUnlabeledDataset
            return CivilCommentsUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
            return CivilCommentsDataset(version=version, **dataset_kwargs)

    elif dataset == 'domainnet':
        if unlabeled:
            from wilds.datasets.unlabeled.domainnet_unlabeled_dataset import DomainNetUnlabeledDataset
            return DomainNetUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            from wilds.datasets.domainnet_dataset import DomainNetDataset
            return DomainNetDataset(version=version, **dataset_kwargs)

    elif dataset == 'iwildcam':
        if unlabeled:
            from wilds.datasets.unlabeled.iwildcam_unlabeled_dataset import IWildCamUnlabeledDataset
            return IWildCamUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            if version == '1.0':
                from wilds.datasets.archive.iwildcam_v1_0_dataset import IWildCamDataset
            else:
                from wilds.datasets.iwildcam_dataset import IWildCamDataset # type:ignore
            return IWildCamDataset(version=version, **dataset_kwargs)

    elif dataset == 'waterbirds':
        from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
        return WaterbirdsDataset(version=version, **dataset_kwargs)

    elif dataset == 'yelp':
        from wilds.datasets.yelp_dataset import YelpDataset
        return YelpDataset(version=version, **dataset_kwargs)

    elif dataset == 'ogb-molpcba':
        if unlabeled:
            from wilds.datasets.unlabeled.ogbmolpcba_unlabeled_dataset import OGBPCBAUnlabeledDataset
            return OGBPCBAUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            from wilds.datasets.ogbmolpcba_dataset import OGBPCBADataset
            return OGBPCBADataset(version=version, **dataset_kwargs)

    elif dataset == 'poverty':
        if unlabeled:
            from wilds.datasets.unlabeled.poverty_unlabeled_dataset import PovertyMapUnlabeledDataset
            return PovertyMapUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            if version == '1.0':
                from wilds.datasets.archive.poverty_v1_0_dataset import PovertyMapDataset
            else:            
                from wilds.datasets.poverty_dataset import PovertyMapDataset # type:ignore
            return PovertyMapDataset(version=version, **dataset_kwargs)

    elif dataset == 'fmow':
        if unlabeled:
            from wilds.datasets.unlabeled.fmow_unlabeled_dataset import FMoWUnlabeledDataset
            return FMoWUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            if version == '1.0':
                from wilds.datasets.archive.fmow_v1_0_dataset import FMoWDataset
            else:
                from wilds.datasets.fmow_dataset import FMoWDataset # type:ignore
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
        
    elif dataset == 'globalwheat':
        if unlabeled:
            from wilds.datasets.unlabeled.globalwheat_unlabeled_dataset import GlobalWheatUnlabeledDataset
            return GlobalWheatUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            from wilds.datasets.globalwheat_dataset import GlobalWheatDataset # type:ignore
            return GlobalWheatDataset(version=version, **dataset_kwargs)

    elif dataset == 'encode':
        from wilds.datasets.encode_dataset import EncodeDataset
        return EncodeDataset(version=version, **dataset_kwargs)

    elif dataset == 'rxrx1':
        from wilds.datasets.rxrx1_dataset import RxRx1Dataset
        return RxRx1Dataset(version=version, **dataset_kwargs)

    elif dataset == 'globalwheat':
        from wilds.datasets.globalwheat_dataset import GlobalWheatDataset
        return GlobalWheatDataset(version=version, **dataset_kwargs)
