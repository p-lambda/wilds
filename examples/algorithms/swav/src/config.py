#############################################
# SwAV-specific defaults for WILDS Datasets #
#############################################

# Maximum batch size that fits on a 12GB GPU
MAX_BATCH_SIZE_PER_GPU = {
    "camelyon17": 168,
    "iwildcam": 24,
    "fmow": 72,
    "poverty": 120,
}

def get_base_lr(dataset, gpus=4):
    # base_lr= DEFAULT_LR / (DEFAULT_BATCH_SIZE / $effective_batch_size),
    # where DEFAULT_LR=4.8, DEFAULT_BATCH_SIZE=4096 and effective_batch_size=batch size per gpu * $NUM_GPUS.
    # base_lr= 4.8 / (4096 / $effective_batch_size).
    batch_size_per_gpu = MAX_BATCH_SIZE_PER_GPU[dataset]
    effective_batch_size = batch_size_per_gpu * gpus
    return 4.8 / (4096. / effective_batch_size)


# All the defaults are configured to run on 4 GPUs.
DATASET_DEFAULTS = {
    'camelyon17': {
        'split_scheme': 'official',
        'model': 'densenet121',
        'model_kwargs': {},
        'train_transform': 'image_base',
        'eval_transform': 'image_base',
        'target_resolution': (96, 96),
        'loss_function': 'cross_entropy',
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'scheduler': None,
        'batch_size': MAX_BATCH_SIZE_PER_GPU['camelyon17'],
        'lr': get_base_lr('camelyon17'),
        'final_lr': get_base_lr('camelyon17') / 1000.,
        'weight_decay': 0.01,
        'epsilon': 0.03,
        'nmb_prototypes': 20,
        'queue_length': 3840,   # TODO: we need to change this depending on the batch size
        'epoch_queue_starts': 500,
        'warmup_epochs': 0,
        'n_epochs': 400,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
        'loader_kwargs': {
            'num_workers': 8,
            'pin_memory': True,
            'drop_last': True,
        },
    },
    'domainnet': {
        'split_scheme': 'official',
        'dataset_kwargs': {
            'source_domain': 'sketch',
            'target_domain': 'real',
            'use_sentry': True,
        },
        'model': 'resnet50',
        'model_kwargs': {},
        'train_transform': 'image_resize_and_center_crop',
        'eval_transform': 'image_resize_and_center_crop',
        'resize_scale': 256.0 / 224.0,
        'target_resolution': (224, 224),
        'loss_function': 'cross_entropy',
        'batch_size': 128,
        'optimizer': 'SGD',
        'lr': 0.6,
        'final_lr': 0.6 / 1000.,
        'weight_decay': 1e-3,
        'epsilon': 0.03,
        'nmb_prototypes': 400,
        'queue_length': 3840,
        'epoch_queue_starts': 500,
        'warmup_epochs': 0,
        'n_epochs': 400,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
        'loader_kwargs': {
            'num_workers': 8,
            'pin_memory': True,
            'drop_last': True,
        },
    },
    'fmow': {
        'split_scheme': 'official',
        'dataset_kwargs': {
            'seed': 111,
            'use_ood_val': True
        },
        'model': 'densenet121',
        'model_kwargs': {},
        'target_resolution': (224, 224),
        'train_transform': 'image_base',
        'eval_transform': 'image_base',
        'loss_function': 'cross_entropy',
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma': 0.96},
        'batch_size': MAX_BATCH_SIZE_PER_GPU["fmow"],
        'lr': get_base_lr('fmow'),
        'final_lr': get_base_lr('fmow') / 1000.,
        'weight_decay': 0.0,
        'warmup_epochs': 0,
        'epsilon': 0.03,
        'nmb_prototypes': 620,
        'queue_length': 3840,
        'epoch_queue_starts': 500,
        'n_epochs': 400,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
        'loader_kwargs': {
            'num_workers': 8,
            'pin_memory': True,
            'drop_last': True,
        },
    },
    'iwildcam': {
        'loss_function': 'cross_entropy',
        'val_metric': 'F1-macro_all',
        'train_transform': 'image_base',
        'eval_transform': 'image_base',
        'target_resolution': (448, 448),
        'model': 'resnet50',
        'model_kwargs': {},
        'lr': get_base_lr('iwildcam'),
        'final_lr': get_base_lr('iwildcam') / 1000.,
        'weight_decay': 0.0,
        'batch_size': MAX_BATCH_SIZE_PER_GPU["iwildcam"],
        'warmup_epochs': 0,
        'epsilon': 0.03,
        'nmb_prototypes': 1860,
        'queue_length': 3840,
        'epoch_queue_starts': 500,
        'n_epochs': 400,
        'optimizer': 'Adam',
        'split_scheme': 'official',
        'scheduler': None,
        'groupby_fields': ['location'],
        'no_group_logging': True,
        'process_outputs_function': 'multiclass_logits_to_pred',
        'loader_kwargs': {
            'num_workers': 8,
            'pin_memory': True,
            'drop_last': True,
        },
    },
    'poverty': {
        'split_scheme': 'official',
        'dataset_kwargs': {
            'no_nl': False,
            'fold': 'A',
            'use_ood_val': True
        },
        'model': 'resnet18_ms',
        'model_kwargs': {'num_channels': 8},
        'train_transform': 'poverty_train',
        'eval_transform': None,
        'target_resolution': (224, 224),
        'loss_function': 'mse',
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma': 0.96},
        'batch_size': MAX_BATCH_SIZE_PER_GPU["poverty"],
        'lr': get_base_lr('poverty'),
        'final_lr': get_base_lr('poverty') / 1000.,
        'weight_decay': 0.0,
        'epsilon': 0.03,
        'nmb_prototypes': 10,
        'queue_length': 3840,
        'epoch_queue_starts': 500,
        'warmup_epochs': 0,
        'n_epochs': 400,
        'process_outputs_function': None,
        'loader_kwargs': {
            'num_workers': 8,
            'pin_memory': True,
            'drop_last': True,
        },
    }
}