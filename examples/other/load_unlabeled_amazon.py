"""
TODO: Don't include this file later. Turn this into documentation:
      https://github.com/p-lambda/wilds-unlabeled/issues/2
"""

import pdb

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

dataset = get_dataset(dataset="amazon", unlabeled=True, download=True)
unlabeled_data = dataset.get_subset("extra_unlabeled")
data_loader = get_train_loader("standard", unlabeled_data, batch_size=16)

for x, metadata in data_loader:
    pdb.set_trace()
    print(x)
