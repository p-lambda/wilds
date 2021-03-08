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
]

additional_datasets = [
    'celebA',
    'waterbirds',
    'yelp',
    'bdd100k',
    'sqf',
]

supported_datasets = benchmark_datasets + additional_datasets
