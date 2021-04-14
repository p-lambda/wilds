"""
TODO: don't include this file later. Turn this into documentation. -Tony
"""

import pdb

from wilds import get_dataset
from wilds.common.data_loaders import get_unlabeled_loader

dataset = get_dataset(dataset="amazon", download=True)
unlabeled_data = dataset.get_subset("ood_unlabeled")
data_loader = get_unlabeled_loader("standard", unlabeled_data, batch_size=16)

for x, metadata in data_loader:
    pdb.set_trace()
    print(x)
