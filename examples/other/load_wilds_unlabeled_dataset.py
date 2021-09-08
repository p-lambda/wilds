"""
TODO: Don't include this file later. Turn this into documentation:
      https://github.com/p-lambda/wilds-unlabeled/issues/2
"""

import pdb

from tqdm import tqdm

from examples.utils import InfiniteDataIterator
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

WILDS_DATASET = "civilcomments"

# Labeled data
dataset = get_dataset(dataset=WILDS_DATASET, download=True, root_dir="../data")
labeled_subset = dataset.get_subset("train")
labeled_data_loader = get_train_loader("standard", labeled_subset, batch_size=2)

# Unlabeled data
dataset = get_dataset(
    dataset=WILDS_DATASET, unlabeled=True, download=True, root_dir="../data"
)
unlabeled_subset = dataset.get_subset("extra_unlabeled")
unlabeled_data_loader = get_train_loader("standard", unlabeled_subset, batch_size=32)
unlabeled_data_iterator = InfiniteDataIterator(unlabeled_data_loader)

batch = 0
for labeled_batch in tqdm(labeled_data_loader):
    if batch % 1000 == 0:
        print(f"Batch #{batch}")
    unlabeled_batch = next(unlabeled_data_iterator)
    batch += 1
