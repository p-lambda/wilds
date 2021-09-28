import copy
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast

from data_augmentation.randaugment import FIX_MATCH_AUGMENTATION_POOL, RandAugment


_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


def initialize_transform(
    transform_name, config, dataset, is_training, additional_transform_name=None
):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.
    """
    if transform_name is None:
        return None
    elif transform_name == "bert":
        return initialize_bert_transform(config)
    elif transform_name == 'rxrx1':
        return initialize_rxrx1_transform(is_training)

    # For images
    normalize = True
    if transform_name == "image_base":
        transform_steps = get_image_base_transform_steps(config, dataset)
    elif transform_name == "image_resize":
        transform_steps = get_image_resize_transform_steps(
            config, dataset
        )
    elif transform_name == "image_resize_and_center_crop":
        transform_steps = get_image_resize_and_center_crop_transform_steps(
            config, dataset
        )
    elif transform_name == "poverty":
        if not is_training:
            return None
        transform_steps = []
        normalize = False
    else:
        raise ValueError(f"{transform_name} not recognized")

    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )
    if additional_transform_name == "fixmatch":
        transformations = add_fixmatch_transform(
            config, dataset, transform_steps, default_normalization
        )
        transform = MultipleTransforms(transformations)
    elif additional_transform_name == "randaugment":
        if transform_name == 'poverty':
            transform = add_poverty_rand_augment_transform(
                config, dataset, transform_steps
            )
        else:
            transform = add_rand_augment_transform(
                config, dataset, transform_steps, default_normalization
            )
    elif additional_transform_name == "weak":
        transform = add_weak_transform(
            config, dataset, transform_steps, default_normalization
        )
    else:
        if transform_name != "poverty":
            # The poverty data is already a tensor at this point
            transform_steps.append(transforms.ToTensor())
        if normalize:
            transform_steps.append(default_normalization)
        transform = transforms.Compose(transform_steps)

    return transform


def initialize_bert_transform(config):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in config.model
    assert config.max_token_length is not None

    tokenizer = get_bert_tokenizer(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_token_length,
            return_tensors="pt",
        )
        if config.model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif config.model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform

def initialize_rxrx1_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform

def get_image_base_transform_steps(config, dataset) -> List:
    transform_steps = []

    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))

    if config.target_resolution is not None:
        transform_steps.append(transforms.Resize(config.target_resolution))

    return transform_steps


def get_image_resize_and_center_crop_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    transform_steps = get_image_resize_transform_steps(config, dataset)
    target_resolution = _get_target_resolution(config, dataset)
    transform_steps.append(
        transforms.CenterCrop(target_resolution),
    )
    return transform_steps


def get_image_resize_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(
        int(res * config.resize_scale) for res in dataset.original_resolution
    )
    return [
        transforms.Resize(scaled_resolution)
    ]

def add_fixmatch_transform(config, dataset, base_transform_steps, normalization):
    return (
        add_weak_transform(config, dataset, base_transform_steps, normalization),
        add_rand_augment_transform(config, dataset, base_transform_steps, normalization)
    )

def add_weak_transform(config, dataset, base_transform_steps, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    weak_transform_steps = copy.deepcopy(base_transform_steps)
    weak_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return transforms.Compose(weak_transform_steps)

def add_rand_augment_transform(config, dataset, base_transform_steps, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution
            ),
            RandAugment(
                n=config.randaugment_n,
                augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return transforms.Compose(strong_transform_steps)

def poverty_rgb_color_transform(ms_img, transform):
    from wilds.datasets.poverty_dataset import _MEANS_2009_17, _STD_DEVS_2009_17
    poverty_rgb_means = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
    poverty_rgb_stds = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))

    def unnormalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result[:3] = (result[:3] * poverty_rgb_stds) + poverty_rgb_means
        return result

    def normalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result[:3] = (result[:3] - poverty_rgb_means) / poverty_rgb_stds
        return ms_img

    color_transform = transforms.Compose([
        transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
        transform,
        transforms.Lambda(lambda ms_img: normalize_rgb_in_poverty_ms_img(ms_img)),
    ])
    # The first three channels of the Poverty MS images are BGR
    # So we shuffle them to the standard RGB to do the ColorJitter
    # Before shuffling them back
    ms_img[:3] = color_transform(ms_img[[2,1,0]])[[2,1,0]] # bgr to rgb to bgr
    return ms_img

def add_poverty_rand_augment_transform(config, dataset, base_transform_steps):
    def poverty_color_jitter(ms_img):
        return poverty_rgb_color_transform(
            ms_img,
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1))

    # def viz(ms_img):
    #     # This function is just to visualize the images for exploratory/debugging purposes
    #     from wilds.datasets.poverty_dataset import _MEANS_2009_17, _STD_DEVS_2009_17
    #     poverty_rgb_means = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
    #     poverty_rgb_stds = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
    #     def unnormalize_rgb_in_poverty_ms_img(ms_img):
    #         ms_img[:3] = (ms_img[:3] * poverty_rgb_stds) + poverty_rgb_means
    #         return ms_img
    #     color_transform = transforms.Compose([
    #         transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img))
    #     ])
    #     ms_img[:3] = color_transform(ms_img[[2,1,0]])[[2,1,0]] # bgr to rgb to bgr
    #     return ms_img

    def ms_cutout(ms_img):
        def _sample_uniform(a, b):
            return torch.empty(1).uniform_(a, b).item()

        assert ms_img.shape[1] == ms_img.shape[2]
        img_width = ms_img.shape[1]
        cutout_width = _sample_uniform(0, img_width/2)
        cutout_center_x = _sample_uniform(0, img_width)
        cutout_center_y = _sample_uniform(0, img_width)
        x0 = int(max(0, cutout_center_x - cutout_width/2))
        y0 = int(max(0, cutout_center_y - cutout_width/2))
        x1 = int(min(img_width, cutout_center_x + cutout_width/2))
        y1 = int(min(img_width, cutout_center_y + cutout_width/2))

        # Fill with 0 because the data is already normalized to mean zero
        ms_img[:, x0:x1, y0:y1] = 0
        return ms_img

    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
        transforms.Lambda(lambda ms_img: poverty_color_jitter(ms_img)),
        transforms.Lambda(lambda ms_img: ms_cutout(ms_img)),
        # transforms.Lambda(lambda ms_img: viz(ms_img)),
    ])

    return transforms.Compose(strong_transform_steps)

def _get_target_resolution(config, dataset):
    if config.target_resolution is not None:
        return config.target_resolution
    else:
        return dataset.original_resolution


class MultipleTransforms(object):
    """When multiple transformations of the same data need to be returned."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transformations)
