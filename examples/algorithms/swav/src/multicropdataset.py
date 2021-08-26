# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter, Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from wilds import get_dataset
# In order to get the examples module in the path
import sys
sys.path.insert(0, sys.path[0] + '/../../..')
from examples.transforms import initialize_transform

logger = getLogger()


class CustomSplitDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        root_dir,
        config,
    ):
        super().__init__()

        self.datasets = []
        dataset = get_dataset(
            dataset=dataset_name,
            root_dir=root_dir,
            unlabeled=True,
            download=True,
            **config.dataset_kwargs
        )
        train_transform = initialize_transform(
            transform_name=config.train_transform,
            config=config,
            dataset=dataset,
        )
        for split in config.splits:
            subset = dataset.get_subset(split, transform=None)
            self.datasets.append(subset)
        self.dataset_lengths = [len(d) for d in self.datasets]

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, index):
        # determine from which dataset to take this
        ds_idx = 0
        while ds_idx < len(self.dataset_lengths):
            if index < self.dataset_lengths[ds_idx]:
                break
            index -= self.dataset_lengths[ds_idx]
            ds_idx += 1
        # ds_idx now stores the correct dataset, and index stores
        # the correct position within that dataset
        x, _ = self.datasets[ds_idx][index] # discard metadata
        return x


class CustomSplitMultiCropDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        root_dir,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        config,
        return_index=False
    ):
        super().__init__()

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.return_index = return_index

        self.ds = CustomSplitDataset(dataset_name, root_dir, config)
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        trans = []
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=means, std=stds)])
            ] * nmb_crops[i])
        self.trans = trans

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image = self.ds[index]
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
