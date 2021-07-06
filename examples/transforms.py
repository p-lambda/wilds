import torchvision.transforms as transforms
import torch
from data_augmentation.randaugment import FIX_MATCH_AUGMENTATION_POOL, RandAugment
from transformers import BertTokenizerFast, DistilBertTokenizerFast

def initialize_transform(transform_name, config, dataset):
    if transform_name is None:
        return None
    elif transform_name == 'bert':
        return initialize_bert_transform(config)
    elif transform_name == 'image_base':
        return initialize_image_base_transform(config, dataset)
    elif transform_name == 'image_resize_and_center_crop':
        return initialize_image_resize_and_center_crop_transform(config, dataset)
    elif transform_name == 'fix_match':
        return initialize_fix_match_transform(config)
    elif transform_name == 'poverty_train':
        return initialize_poverty_train_transform()
    else:
        raise ValueError(f"{transform_name} not recognized")

    return transform

def initialize_bert_transform(config):
    assert 'bert' in config.model
    assert config.max_token_length is not None

    tokenizer = getBertTokenizer(config.model)
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.max_token_length,
            return_tensors='pt')
        if config.model == 'bert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        elif config.model == 'distilbert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform

def getBertTokenizer(model):
    if model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f'Model: {model} not recognized.')

    return tokenizer

def initialize_image_base_transform(config, dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(dataset.original_resolution)!=max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    if config.target_resolution is not None and config.dataset!='fmow':
        transform_steps.append(transforms.Resize(config.target_resolution))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform = transforms.Compose(transform_steps)
    return transform

def initialize_image_resize_and_center_crop_transform(config, dataset):
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(int(res*config.resize_scale) for res in dataset.original_resolution)
    if config.target_resolution is not None:
        target_resolution = config.target_resolution
    else:
        target_resolution = dataset.original_resolution
    transform = transforms.Compose([
        transforms.Resize(scaled_resolution),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def initialize_poverty_train_transform():
    transforms_ls = [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
        transforms.ToTensor()]
    rgb_transform = transforms.Compose(transforms_ls)

    def transform_rgb(img):
        # bgr to rgb and back to bgr
        img[:3] = rgb_transform(img[:3][[2,1,0]])[[2,1,0]]
        return img
    transform = transforms.Lambda(lambda x: transform_rgb(x))
    return transform


def initialize_fix_match_transform(config):
    # TODO: hardcoded mean and std. How do we determine a good mean and std? hyperparameter tuning? -Tony
    return TransformFixMatch(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761],
        randaugment_n=config.randaugment_n,
        randaugment_m=config.randaugment_m,
        augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
    )

class TransformFixMatch(object):
    # Adapted from https://github.com/kekmodel/FixMatch-pytorch
    def __init__(self, mean, std, randaugment_n, randaugment_m, augmentation_pool):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32,
                    padding=int(32 * 0.125),
                    padding_mode='reflect'
                )
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32,
                    padding=int(32 * 0.125),
                    padding_mode='reflect'
                ),
                # TODO: Support text RandAugment later.
                #       Then, create a factory that returns either the image or text RandAugment -Tony
                RandAugment(
                    n=randaugment_n,
                    m=randaugment_m,
                    augmentation_pool=augmentation_pool,
                )
            ]
        )
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)