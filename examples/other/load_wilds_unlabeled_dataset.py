"""
TODO: Don't include this file later. Turn this into documentation:
      https://github.com/p-lambda/wilds-unlabeled/issues/2
"""

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

# Load the labeled data
dataset = get_dataset(dataset="fmow", download=True)
labeled_subset = dataset.get_subset("train", transform=transforms.ToTensor())
data_loader = get_train_loader("standard", labeled_subset, batch_size=16)

# Load the unlabeled data
unlabeled_dataset = get_dataset(dataset="fmow", unlabeled=True, download=True)
unlabeled_subset = unlabeled_dataset.get_subset("test_unlabeled", transform=transforms.ToTensor())
unlabeled_data_loader = get_train_loader("standard", unlabeled_subset, batch_size=64)

# Train loop
for labeled_batch, unlabeled_batch in zip(data_loader, unlabeled_data_loader):
    x, y, metadata = labeled_batch
    unlabeled_x, unlabeled_metadata = unlabeled_batch
    ...