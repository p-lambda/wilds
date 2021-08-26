"""
TODO: Don't include this file later. Turn this into documentation:
      https://github.com/p-lambda/wilds-unlabeled/issues/2
"""

import pdb

from tqdm import tqdm

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

# Labeled data
dataset = get_dataset(dataset="amazon", download=True)
labeled_subset = dataset.get_subset("train")
labeled_data_loader = get_train_loader("standard", labeled_subset, batch_size=8)

# Unlabeled data
dataset = get_dataset(dataset="amazon", unlabeled=True, download=True)
unlabeled_subset = dataset.get_subset("extra_unlabeled")
unlabeled_data_loader = get_train_loader("standard", unlabeled_subset, batch_size=16)

for labeled_batch, unlabeled_batch in tqdm(zip(labeled_data_loader, unlabeled_data_loader)):
    pdb.set_trace()
