from .version import __version__
from .get_dataset import get_dataset

benchmark_datasets = [
    'amazon',
    'bdd100k-det',
    'camelyon17',
    'civilcomments',
    'fmow',
    'gwhd',
    'iwildcam',
    'ogb-molpcba',
    'poverty',
    'py150',
]

additional_datasets = [
    'bdd100k-cls',
    'celebA',
    'sqf',
    'waterbirds',
    'yelp',
]

supported_datasets = benchmark_datasets + additional_datasets
