from .version import __version__
from .get_dataset import get_dataset

benchmark_datasets = [
    'amazon',
    'camelyon17',
    'civilcomments',
    'iwildcam',
    'ogb-molpcba',
    'poverty',
    'fmow',
    'py150',
    'rxrx1',
    'globalwheat',
]

additional_datasets = [
    'celebA',
    'domainnet',
    'waterbirds',
    'yelp',
    'bdd100k',
    'sqf',
    'encode',
    'py150aug',
    'py150-mini',
    'py150aug-mini',
]

supported_datasets = benchmark_datasets + additional_datasets

unlabeled_datasets = [
    'amazon',
    'camelyon17',
    'domainnet',
    'civilcomments',
    'iwildcam',
    'ogb-molpcba',
    'poverty',
    'fmow',
    'globalwheat',
]

unlabeled_splits = [
    'train_unlabeled',
    'val_unlabeled',
    'test_unlabeled',
    'extra_unlabeled'
]