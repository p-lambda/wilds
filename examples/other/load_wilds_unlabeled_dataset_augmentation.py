import os
import pdb
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from examples.utils import InfiniteDataIterator
from examples.transforms import initialize_transform
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

WILDS_DATASET = "domainnet"


class Namespace(object):
    def __init__(self, some_dict):
        self.__dict__.update(some_dict)


def display_image(img):
    to_image = transforms.ToPILImage()
    plt.imshow(to_image(img))
    plt.show()


# Labeled data
dataset = get_dataset(dataset=WILDS_DATASET, download=True, root_dir="../data")
train_transform = initialize_transform(
    transform_name="image_base",
    config=Namespace(
        {"dataset": WILDS_DATASET, "target_resolution": (448, 448), "randaugment_n": 2}
    ),
    dataset=dataset,
    additional_transform_name="randaugment",
    is_training=False,
)
labeled_subset = dataset.get_subset("train", transform=train_transform)
labeled_data_loader = get_train_loader("standard", labeled_subset, batch_size=16)

# Unlabeled data
train_transform = initialize_transform(
    transform_name="image_base",
    config=Namespace(
        {"dataset": WILDS_DATASET, "target_resolution": (448, 448), "randaugment_n": 2}
    ),
    dataset=dataset,
    additional_transform_name=None,
    is_training=False,
)
clean_subset = dataset.get_subset("train", transform=train_transform)
clean_data_loader = get_train_loader("standard", clean_subset, batch_size=16)
clean_data_iterator = InfiniteDataIterator(clean_data_loader)

batch = 0
for labeled_batch in tqdm(labeled_data_loader):
    x, _, metadata = labeled_batch
    clean_batch = next(clean_data_iterator)
    x_clean, _, metadata_clean = clean_batch
    display_image(x[0])
    display_image(x_clean[0])
    pdb.set_trace()
    batch += 1
    break
