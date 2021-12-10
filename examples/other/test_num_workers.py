import argparse
import pdb
import os
import sys
import time

from tqdm import tqdm

# TODO: This is needed to test the WILDS package locally. Remove later -Tony
sys.path.insert(1, os.path.join(sys.path[0], "../.."))

from examples.utils import InfiniteDataIterator
from examples.transforms import initialize_transform
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

WILDS_DATASET = "iwildcam"
TRANSFORM_NAME = "image_base"
NUM_WORKERS_TO_TRY = [2, 4, 6, 8, 16, 32]

"""
Usage: 
Run in CodaLab: python3 wilds-unlabeled/examples/other/test_num_workers.py data
Run locally:    python3 examples/other/test_num_workers.py <root_dir>
"""


class Namespace(object):
    def __init__(self, some_dict):
        self.__dict__.update(some_dict)


def main():
    for labeled_num_workers in NUM_WORKERS_TO_TRY:
        for unlabeled_num_workers in NUM_WORKERS_TO_TRY:
            start_time = time.time()
            # Labeled data
            dataset = get_dataset(dataset=WILDS_DATASET, root_dir=args.root_dir)
            train_transform = initialize_transform(
                transform_name=TRANSFORM_NAME,
                config=Namespace({"dataset": WILDS_DATASET, "target_resolution": (448, 448), "randaugment_n": 2}),
                dataset=dataset,
                additional_transform_name="randaugment",
                is_training=True,
            )
            labeled_subset = dataset.get_subset("train", transform=train_transform)
            labeled_data_loader = get_train_loader(
                "standard",
                labeled_subset,
                batch_size=6,
                **{"num_workers": labeled_num_workers, "pin_memory": True},
            )

            # Unlabeled data
            dataset = get_dataset(dataset=WILDS_DATASET, unlabeled=True, root_dir=args.root_dir)
            unlabeled_subset = dataset.get_subset(
                "extra_unlabeled", transform=train_transform
            )
            unlabeled_data_loader = get_train_loader(
                "standard",
                unlabeled_subset,
                batch_size=42,
                **{"num_workers": unlabeled_num_workers, "pin_memory": True},
            )
            unlabeled_data_iterator = InfiniteDataIterator(unlabeled_data_loader)

            start_time_first_iteration = time.time()
            for labeled_batch in tqdm(labeled_data_loader):
                unlabeled_batch = next(unlabeled_data_iterator)
            end_time_first_iteration = time.time()
            print(f"first iteration time (minutes): {(end_time_first_iteration - start_time_first_iteration) / 60}")
            for labeled_batch in tqdm(labeled_data_loader):
                unlabeled_batch = next(unlabeled_data_iterator)
            total_time_seconds = time.time() - start_time
            print(
                f"num_workers={labeled_num_workers}, unlabeled_num_workers={unlabeled_num_workers} "
                f"total time (minutes): {total_time_seconds / 60}"
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different num_workers.")
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to dataset.",
    )
    # Parse args and run this script
    args = parser.parse_args()
    print(args)
    main()
