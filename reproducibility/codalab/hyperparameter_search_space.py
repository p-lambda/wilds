import math

from examples.configs.datasets import dataset_defaults

AMAZON = "amazon"
CIVIL_COMMENTS = "civilcomments"
CAMELYON17 = "camelyon17"
IWILDCAM = "iwildcam"
FMOW = "fmow"
POVERTY = "poverty"

# Maximum batch size that fits on a 12GB GPU
MAX_BATCH_SIZES = {
    AMAZON: 24,
    CIVIL_COMMENTS: 48,
    CAMELYON17: 168,
    IWILDCAM: 24,
    FMOW: 72,
    POVERTY: 120,
}

DEFAULT_UNLABELED_FRAC = [3 / 4, 7 / 8, 15 / 16]

NOISY_STUDENT_TEACHERS = {
    CAMELYON17: "",
    IWILDCAM: "0x52f2dd8e448a4c7e802783fa35c269c6",
    FMOW: "",
    POVERTY: "",
}


def get_epochs_unlabeled(dataset, factor=1, parts=[4, 8, 16]):
    default_n_epochs = dataset_defaults[dataset]["n_epochs"]
    return [math.ceil((default_n_epochs * factor) / part) for part in parts]


def get_lr_grid(dataset, grad_accumulation=1):
    default_lr = dataset_defaults[dataset]["lr"]
    default_batch_size = dataset_defaults[dataset]["batch_size"]
    max_batch_size = MAX_BATCH_SIZES[dataset]
    new_lr = default_lr * ((max_batch_size * grad_accumulation) / default_batch_size)
    # We sample a value 10^U(a, b)
    return [math.log10(new_lr / 10), math.log10(new_lr * 10)]


ERM_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        AMAZON: {
            "batch_size": [MAX_BATCH_SIZES[AMAZON]],
            "lr": get_lr_grid(AMAZON, grad_accumulation=1),
        },
        CIVIL_COMMENTS: {
            "batch_size": [MAX_BATCH_SIZES[CIVIL_COMMENTS]],
            "lr": get_lr_grid(CIVIL_COMMENTS, grad_accumulation=1),
        },
        CAMELYON17: {
            "batch_size": [MAX_BATCH_SIZES[CAMELYON17]],
            "lr": get_lr_grid(CAMELYON17, grad_accumulation=1),
        },
        IWILDCAM: {
            "batch_size": [MAX_BATCH_SIZES[IWILDCAM]],
            "lr": get_lr_grid(IWILDCAM, grad_accumulation=1),
        },
        FMOW: {
            "batch_size": [MAX_BATCH_SIZES[FMOW]],
            "lr": get_lr_grid(FMOW, grad_accumulation=1),
        },
        POVERTY: {
            "batch_size": [MAX_BATCH_SIZES[POVERTY]],
            "lr": get_lr_grid(POVERTY, grad_accumulation=1),
        },
    },
}

ERM_AUGMENT_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        CAMELYON17: {
            "batch_size": [MAX_BATCH_SIZES[CAMELYON17]],
            "lr": get_lr_grid(CAMELYON17, grad_accumulation=1),
            "additional_train_transform": ["randaugment"],
        },
        IWILDCAM: {
            "batch_size": [MAX_BATCH_SIZES[IWILDCAM]],
            "lr": get_lr_grid(IWILDCAM, grad_accumulation=1),
            "additional_train_transform": ["randaugment"],
        },
        FMOW: {
            "batch_size": [MAX_BATCH_SIZES[FMOW]],
            "lr": get_lr_grid(FMOW, grad_accumulation=1),
            "additional_train_transform": ["randaugment"],
        },
        POVERTY: {
            "batch_size": [MAX_BATCH_SIZES[POVERTY]],
            "lr": get_lr_grid(POVERTY, grad_accumulation=1),
            "additional_train_transform": ["randaugment"],
        },
    },
}

CORAL_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        AMAZON: {
            "lr": get_lr_grid(AMAZON, grad_accumulation=4),
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(AMAZON, factor=2),
        },
        CIVIL_COMMENTS: {
            "lr": get_lr_grid(CIVIL_COMMENTS, grad_accumulation=4),
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(CIVIL_COMMENTS, factor=2),
        },
        CAMELYON17: {
            "lr": get_lr_grid(CAMELYON17, grad_accumulation=4),
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(CAMELYON17, factor=2),
        },
        IWILDCAM: {
            "lr": get_lr_grid(IWILDCAM, grad_accumulation=4),
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(IWILDCAM, factor=2),
        },
        FMOW: {
            "lr": get_lr_grid(FMOW, grad_accumulation=4),
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(FMOW, factor=2),
        },
        POVERTY: {
            "lr": get_lr_grid(POVERTY, grad_accumulation=4),
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(POVERTY, factor=2),
        },
    },
}

DANN_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        AMAZON: {
            "dann_classifier_lr": get_lr_grid(AMAZON, grad_accumulation=4),
            "dann_discriminator_lr": get_lr_grid(AMAZON, grad_accumulation=4),
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(AMAZON, factor=2),
        },
        CIVIL_COMMENTS: {
            "dann_classifier_lr": get_lr_grid(CIVIL_COMMENTS, grad_accumulation=4),
            "dann_discriminator_lr": get_lr_grid(CIVIL_COMMENTS, grad_accumulation=4),
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(CIVIL_COMMENTS, factor=2),
        },
        CAMELYON17: {
            "dann_classifier_lr": get_lr_grid(CAMELYON17, grad_accumulation=4),
            "dann_discriminator_lr": get_lr_grid(CAMELYON17, grad_accumulation=4),
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(CAMELYON17, factor=2),
        },
        IWILDCAM: {
            "dann_classifier_lr": get_lr_grid(IWILDCAM, grad_accumulation=4),
            "dann_discriminator_lr": get_lr_grid(IWILDCAM, grad_accumulation=4),
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(IWILDCAM, factor=2),
        },
        FMOW: {
            "dann_classifier_lr": get_lr_grid(FMOW, grad_accumulation=4),
            "dann_discriminator_lr": get_lr_grid(FMOW, grad_accumulation=4),
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(FMOW, factor=2),
        },
        POVERTY: {
            "dann_classifier_lr": get_lr_grid(POVERTY, grad_accumulation=4),
            "dann_discriminator_lr": get_lr_grid(POVERTY, grad_accumulation=4),
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(POVERTY, factor=2),
        },
    },
}

FIXMATCH_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        CAMELYON17: {
            "lr": get_lr_grid(CAMELYON17, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(CAMELYON17, factor=2),
        },
        IWILDCAM: {
            "lr": get_lr_grid(IWILDCAM, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(IWILDCAM, factor=2),
        },
        FMOW: {
            "lr": get_lr_grid(FMOW, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(FMOW, factor=2),
        },
        POVERTY: {
            "lr": get_lr_grid(POVERTY, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(POVERTY, factor=2),
        },
    },
}

PSEUDOLABEL_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        AMAZON: {
            "lr": get_lr_grid(AMAZON, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(AMAZON, factor=2),
        },
        CIVIL_COMMENTS: {
            "lr": get_lr_grid(CIVIL_COMMENTS, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(CIVIL_COMMENTS, factor=2),
        },
        CAMELYON17: {
            "lr": get_lr_grid(CAMELYON17, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(CAMELYON17, factor=2),
        },
        IWILDCAM: {
            "lr": get_lr_grid(IWILDCAM, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(IWILDCAM, factor=2),
        },
        FMOW: {
            "lr": get_lr_grid(FMOW, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(FMOW, factor=2),
        },
        POVERTY: {
            "lr": get_lr_grid(POVERTY, grad_accumulation=4),
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(POVERTY, factor=2),
        },
    },
}

NOISY_STUDENT_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        CAMELYON17: {
            "lr": get_lr_grid(CAMELYON17, grad_accumulation=4),
            "scheduler": ["FixMatchLR"],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(CAMELYON17),
        },
        IWILDCAM: {
            "lr": get_lr_grid(IWILDCAM, grad_accumulation=4),
            "scheduler": ["FixMatchLR"],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(IWILDCAM),
        },
        FMOW: {
            "lr": get_lr_grid(FMOW, grad_accumulation=4),
            "scheduler": ["FixMatchLR"],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(FMOW),
        },
        POVERTY: {
            "lr": get_lr_grid(POVERTY, grad_accumulation=4),
            "scheduler": ["FixMatchLR"],
            "unlabeled_batch_size_frac": DEFAULT_UNLABELED_FRAC,
            "n_epochs": get_epochs_unlabeled(POVERTY),
        },
    },
}
