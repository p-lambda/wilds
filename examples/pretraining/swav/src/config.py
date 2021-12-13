#############################################
# SwAV-specific defaults for WILDS Datasets #
#############################################

# Run SwAV on 4 GPUs
NUM_GPUS = 4

# Maximum batch size that fits on a 12GB GPU
MAX_BATCH_SIZE_PER_GPU = {
    "camelyon17": 168,
    "iwildcam": 24,
    "fmow": 72,
    "poverty": 120,
    "domainnet": 96,
}


def get_base_lr(dataset, gpus=NUM_GPUS):
    # base_lr= DEFAULT_LR / (DEFAULT_BATCH_SIZE / $effective_batch_size),
    # where DEFAULT_LR=4.8, DEFAULT_BATCH_SIZE=4096 and effective_batch_size=batch size per gpu * $NUM_GPUS.
    # base_lr= 4.8 / (4096 / $effective_batch_size).
    batch_size_per_gpu = MAX_BATCH_SIZE_PER_GPU[dataset]
    effective_batch_size = batch_size_per_gpu * gpus
    if effective_batch_size == 256:
        return 0.6
    return 4.8 / (4096.0 / effective_batch_size)


def get_queue_length(dataset, gpus=NUM_GPUS):
    batch_size_per_gpu = MAX_BATCH_SIZE_PER_GPU[dataset]
    effective_batch_size = batch_size_per_gpu * gpus
    return 4096 - effective_batch_size


# All the defaults are configured to run on 4 GPUs.
DATASET_DEFAULTS = {
    "camelyon17": {
        "splits": ["test_unlabeled"],
        "split_scheme": "official",
        "dataset_kwargs": {},
        "model": "densenet121",
        "model_kwargs": {"pretrained": False},
        "train_transform": "image_base",
        "eval_transform": "image_base",
        "target_resolution": (96, 96),
        "nmb_crops": [6],
        "size_crops": [96],
        "min_scale_crops": [0.14],
        "max_scale_crops": [1],
        "loss_function": "cross_entropy",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "scheduler": None,
        "batch_size": MAX_BATCH_SIZE_PER_GPU["camelyon17"],
        "lr": get_base_lr("camelyon17"),
        "final_lr": get_base_lr("camelyon17") / 1000.0,
        "epsilon": 0.03,
        "nmb_prototypes": 20,
        "queue_length": get_queue_length("camelyon17"),
        "epoch_queue_starts": 500,
        "warmup_epochs": 0,
        "n_epochs": 400,
        "algo_log_metric": "accuracy",
        "process_outputs_function": "multiclass_logits_to_pred",
        "loader_kwargs": {
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
        },
    },
    "domainnet": {
        "splits": ["test_unlabeled"],
        "split_scheme": "official",
        "dataset_kwargs": {
            "source_domain": "real",
            "target_domain": "sketch",
            "use_sentry": False,
        },
        "model": "resnet50",
        "model_kwargs": {"pretrained": True},
        "train_transform": "image_resize_and_center_crop",
        "eval_transform": "image_resize_and_center_crop",
        "resize_scale": 256.0 / 224.0,
        "target_resolution": (224, 224),
        "nmb_crops": [2, 6],
        "size_crops": [224, 96],
        "min_scale_crops": [0.14, 0.05],
        "max_scale_crops": [1, 0.14],
        "loss_function": "cross_entropy",
        "batch_size": MAX_BATCH_SIZE_PER_GPU["domainnet"],
        "optimizer": "SGD",
        "lr": get_base_lr("domainnet"),
        "final_lr": get_base_lr("domainnet") / 1000.0,
        "epsilon": 0.03,
        "nmb_prototypes": 3450,
        "queue_length": get_queue_length("domainnet"),
        "epoch_queue_starts": 500,
        "warmup_epochs": 0,
        "n_epochs": 400,
        "algo_log_metric": "accuracy",
        "process_outputs_function": "multiclass_logits_to_pred",
        "loader_kwargs": {
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
        },
    },
    "fmow": {
        "splits": ["test_unlabeled"],
        "split_scheme": "official",
        "dataset_kwargs": {"seed": 111, "use_ood_val": True},
        "model": "densenet121",
        "model_kwargs": {"pretrained": True},
        "target_resolution": (224, 224),
        "nmb_crops": [2, 6],
        "size_crops": [224, 96],
        "min_scale_crops": [0.14, 0.05],
        "max_scale_crops": [1, 0.14],
        "train_transform": "image_base",
        "eval_transform": "image_base",
        "loss_function": "cross_entropy",
        "optimizer": "Adam",
        "scheduler": "StepLR",
        "batch_size": MAX_BATCH_SIZE_PER_GPU["fmow"],
        "lr": get_base_lr("fmow"),
        "final_lr": get_base_lr("fmow") / 1000.0,
        "warmup_epochs": 0,
        "epsilon": 0.03,
        "nmb_prototypes": 620,
        "queue_length": get_queue_length("fmow"),
        "epoch_queue_starts": 500,
        "n_epochs": 400,
        "algo_log_metric": "accuracy",
        "process_outputs_function": "multiclass_logits_to_pred",
        "loader_kwargs": {
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
        },
    },
    "iwildcam": {
        "splits": ["extra_unlabeled"],
        "dataset_kwargs": {},
        "loss_function": "cross_entropy",
        "val_metric": "F1-macro_all",
        "train_transform": "image_base",
        "eval_transform": "image_base",
        "target_resolution": (448, 448),
        "nmb_crops": [2, 2],
        "size_crops": [448, 96],
        "min_scale_crops": [0.14, 0.05],
        "max_scale_crops": [1, 0.14],
        "model": "resnet50",
        "model_kwargs": {"pretrained": True},
        "lr": get_base_lr("iwildcam"),
        "final_lr": get_base_lr("iwildcam") / 1000.0,
        "batch_size": MAX_BATCH_SIZE_PER_GPU["iwildcam"],
        "warmup_epochs": 0,
        "epsilon": 0.03,
        "nmb_prototypes": 1860,
        "queue_length": get_queue_length("iwildcam"),
        "epoch_queue_starts": 500,
        "n_epochs": 400,
        "optimizer": "Adam",
        "split_scheme": "official",
        "scheduler": None,
        "groupby_fields": ["location"],
        "no_group_logging": True,
        "process_outputs_function": "multiclass_logits_to_pred",
        "loader_kwargs": {
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
        },
    },
    "poverty": {
        "splits": ["test_unlabeled"],
        "split_scheme": "official",
        "dataset_kwargs": {"no_nl": False, "fold": "A", "use_ood_val": True},
        "model": "resnet18_ms",
        "model_kwargs": {"num_channels": 8},
        "train_transform": "poverty_train",
        "eval_transform": None,
        "target_resolution": (224, 224),
        "nmb_crops": [2, 6],
        "size_crops": [224, 96],
        "min_scale_crops": [0.14, 0.05],
        "max_scale_crops": [1, 0.14],
        "loss_function": "mse",
        "optimizer": "Adam",
        "scheduler": "StepLR",
        "batch_size": MAX_BATCH_SIZE_PER_GPU["poverty"],
        "lr": get_base_lr("poverty"),
        "final_lr": get_base_lr("poverty") / 1000.0,
        "epsilon": 0.03,
        "nmb_prototypes": 1000,
        "queue_length": get_queue_length("poverty"),
        "epoch_queue_starts": 500,
        "warmup_epochs": 0,
        "n_epochs": 400,
        "process_outputs_function": None,
        "loader_kwargs": {
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
        },
    },
}
