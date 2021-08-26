"""
TODO: Don't include this file later. Turn this into documentation:
      https://github.com/p-lambda/wilds-unlabeled/issues/2
"""

import pdb

from niacin.augment import RandAugment
from niacin.text import en
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

# Initialize data augmentor - RandAugment for text. Credit: https://github.com/deniederhut/niacin
# n is the number of transformations to apply sequentially
# m is the order of magnitude. Possible values range from [0, 100].
augmentor = RandAugment([
    # en.add_synonyms,
    # en.add_hyponyms,
    en.add_misspelling,
    # en.swap_words,
    # en.add_contractions,
    # en.add_whitespace,
], n=1, m=15, shuffle=False)

for labeled_batch, unlabeled_batch in tqdm(zip(labeled_data_loader, unlabeled_data_loader)):
    pdb.set_trace()
