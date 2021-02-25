import os, sys
import argparse
import configs.supported as supported

def main():
    """
    Downloads the latest versions of all specified datasets,
    if they do not already exist.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help=f'Specify a space-separated list of dataset names to download. If left unspecified, the script will download all of the official benchmark datasets. Available choices are {list(supported.datasets.keys())}.')
    config = parser.parse_args()

    if config.datasets is None:
        config.datasets = supported.benchmark_datasets

    for dataset in config.datasets:
        if dataset not in supported.datasets:
            raise ValueError(f'{dataset} not recognized; must be one of {list(supported.datasets.keys())}.')

    print(f'Downloading the following datasets: {config.datasets}')
    for dataset in config.datasets:
        print(f'=== {dataset} ===')
        constructor = supported.datasets[dataset]
        constructor(root_dir=config.root_dir, download=True)


if __name__=='__main__':
    main()
