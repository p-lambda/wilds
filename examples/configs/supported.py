import torch.nn as nn
import torch
import sys, os
# Datasets
from wilds.datasets.amazon_dataset import AmazonDataset
from wilds.datasets.bdd100k_dataset import BDD100KDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.celebA_dataset import CelebADataset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.ogbmolpcba_dataset import OGBPCBADataset
from wilds.datasets.poverty_dataset import PovertyMapDataset
from wilds.datasets.sqf_dataset import SQFDataset
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.datasets.yelp_dataset import YelpDataset
from wilds.datasets.py150_dataset import Py150Dataset
# metrics
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import Accuracy, MultiTaskAccuracy, MSE

benchmark_datasets = [
    'amazon',
    'camelyon17',
    'civilcomments',
    'iwildcam',
    'ogb-molpcba',
    'poverty',
    'fmow',
    'py150']

datasets = {
    'amazon': AmazonDataset,
    'camelyon17': Camelyon17Dataset,
    'celebA': CelebADataset,
    'civilcomments': CivilCommentsDataset,
    'iwildcam': IWildCamDataset,
    'waterbirds': WaterbirdsDataset,
    'yelp': YelpDataset,
    'ogb-molpcba': OGBPCBADataset,
    'poverty': PovertyMapDataset,
    'fmow': FMoWDataset,
    'bdd100k': BDD100KDataset,
    'py150': Py150Dataset,
    'sqf': SQFDataset,
}

losses = {
    'cross_entropy': ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none')),
    'lm_cross_entropy': MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none')),
    'mse': MSE(name='loss'),
    'multitask_bce': MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none')),
}

algo_log_metrics = {
    'accuracy': Accuracy(),
    'mse': MSE(),
    'multitask_accuracy': MultiTaskAccuracy(),
    None: None,
}

# see initialize_*() functions for correspondence
transforms = ['bert', 'image_base', 'image_resize_and_center_crop', 'poverty_train']
models = ['resnet18_ms', 'resnet50', 'resnet34', 'wideresnet50',
         'densenet121', 'bert-base-uncased', 'gin-virtual',
         'logistic_regression', 'code-gpt-py']
algorithms = ['ERM', 'groupDRO', 'deepCORAL', 'IRM']
optimizers = ['SGD', 'Adam', 'AdamW']
schedulers = ['linear_schedule_with_warmup', 'ReduceLROnPlateau', 'StepLR']
