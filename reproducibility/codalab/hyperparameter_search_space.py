import math

from examples.configs.datasets import dataset_defaults

# Maximum batch size that fits on a 12GB GPU
MAX_BATCH_SIZES = {
    "amazon": 24,
    "civilcomments": 48,
    "camelyon17": 168,
    "iwildcam": 24,
    "fmow": 72,
    "poverty": 120,
}

DEFAULT_AMAZON_EPOCHS = dataset_defaults["amazon"]["n_epochs"]
DEFAULT_CAMELYON17_EPOCHS = dataset_defaults["camelyon17"]["n_epochs"]
DEFAULT_CIVILCOMMENTS_EPOCHS = dataset_defaults["civilcomments"]["n_epochs"]
DEFAULT_FMOW_EPOCHS = dataset_defaults["fmow"]["n_epochs"]
DEFAULT_IWILDCAM_EPOCHS = dataset_defaults["iwildcam"]["n_epochs"]
DEFAULT_POVERTY_EPOCHS = dataset_defaults["poverty"]["n_epochs"]

def get_epochs_unlabeled(default_n_epochs, parts=[4, 8, 16]):
    return [math.ceil(default_n_epochs / part) for part in parts]

ERM_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "batch_size": [MAX_BATCH_SIZES["amazon"]],
            "lr": [-6, -4],
        },
        "civilcomments": {
            "batch_size": [MAX_BATCH_SIZES["civilcomments"]],
            "lr": [-6, -4],
        },
        "camelyon17": {
            "batch_size": [MAX_BATCH_SIZES["camelyon17"]],
            "lr": [-4, -2],
        },
        "iwildcam": {
            "batch_size": [MAX_BATCH_SIZES["iwildcam"]],
            "lr": [-6, -4],
        },
        "fmow": {
            "batch_size": [MAX_BATCH_SIZES["fmow"]],
            "lr": [-5, -3],
        },
        "poverty": {
            "batch_size": [MAX_BATCH_SIZES["poverty"]],
            "lr": [-4, -2],
        },
    },
}

ERM_AUGMENT_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "camelyon17": {
            "batch_size": [MAX_BATCH_SIZES["camelyon17"]],
            "lr": [-4, -2],
            "additional_train_transform": ["randaugment"],
        },
        "iwildcam": {
            "batch_size": [MAX_BATCH_SIZES["iwildcam"]],
            "lr": [-5, -4],
            "additional_train_transform": ["randaugment"],
        },
        "fmow": {
            "batch_size": [MAX_BATCH_SIZES["fmow"]],
            "lr": [-5, -3],
            "additional_train_transform": ["randaugment"],
        },
        "poverty": {
            "batch_size": [MAX_BATCH_SIZES["poverty"]],
            "lr": [-4, -2],
            "additional_train_transform": ["randaugment"],
        },
    },
}

CORAL_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "lr": [-6, -4],
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_AMAZON_EPOCHS)
        },
        "civilcomments": {
            "lr": [-6, -4],
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_CIVILCOMMENTS_EPOCHS)
        },
        "camelyon17": {
            "lr": [-4, -2],
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_CAMELYON17_EPOCHS)
        },
        "iwildcam": {
            "lr": [-5, -4],
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_IWILDCAM_EPOCHS)
        },
        "fmow": {
            "lr": [-5, -3],
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_FMOW_EPOCHS)
        },
        "poverty": {
            "lr": [-4, -2],
            "coral_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_POVERTY_EPOCHS)
        },
    },
}

DANN_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "dann_classifier_lr": [-6, -4],
            "dann_discriminator_lr": [-6, -4],
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_AMAZON_EPOCHS)
        },
        "civilcomments": {
            "dann_classifier_lr": [-6, -4],
            "dann_discriminator_lr": [-6, -4],
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_CIVILCOMMENTS_EPOCHS)
        },
        "camelyon17": {
            "dann_classifier_lr": [-4, -2],
            "dann_discriminator_lr": [-4, -2],
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_CAMELYON17_EPOCHS)
        },
        "iwildcam": {
            "dann_classifier_lr": [-5, -4],
            "dann_discriminator_lr": [-5, -4],
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_IWILDCAM_EPOCHS)
        },
        "fmow": {
            "dann_classifier_lr": [-5, -3],
            "dann_discriminator_lr": [-5, -3],
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_FMOW_EPOCHS)
        },
        "poverty": {
            "dann_classifier_lr": [-4, -2],
            "dann_discriminator_lr": [-4, -2],
            "dann_penalty_weight": [-1, 1],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_POVERTY_EPOCHS)
        },
    },
}

FIXMATCH_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "camelyon17": {
            "lr": [-4, -2],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(DEFAULT_CAMELYON17_EPOCHS)
        },
        "iwildcam": {
            "lr": [-5, -4],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(DEFAULT_IWILDCAM_EPOCHS),
        },
        "fmow": {
            "lr": [-5, -3],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(DEFAULT_FMOW_EPOCHS),
        },
        "poverty": {
            "lr": [-4, -2],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "scheduler": ["FixMatchLR"],
            "n_epochs": get_epochs_unlabeled(DEFAULT_POVERTY_EPOCHS),
        },
    },
}

PSEUDOLABEL_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "lr": [-6, -4],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_AMAZON_EPOCHS)
        },
        "civilcomments": {
            "lr": [-6, -4],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_CIVILCOMMENTS_EPOCHS)
        },
        "camelyon17": {
            "lr": [-4, -2],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_CAMELYON17_EPOCHS),
        },
        "iwildcam": {
            "lr": [-5, -4],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_IWILDCAM_EPOCHS),
        },
        "fmow": {
            "lr": [-5, -3],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_FMOW_EPOCHS),
        },
        "poverty": {
            "lr": [-4, -2],
            "self_training_lambda": [1],
            "self_training_threshold": [0.7, 0.95],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_POVERTY_EPOCHS),
        },
    },
}

NOISY_STUDENT_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "camelyon17": {
            "lr": [-4, -2],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_CAMELYON17_EPOCHS),
        },
        "iwildcam": {
            "lr": [-5, -4],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_IWILDCAM_EPOCHS),
        },
        "fmow": {
            "lr": [-5, -3],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_FMOW_EPOCHS),
        },
        "poverty": {
            "lr": [-4, -2],
            "unlabeled_batch_size_frac": [3 / 4, 7 / 8, 15 / 16],
            "n_epochs": get_epochs_unlabeled(DEFAULT_POVERTY_EPOCHS),
        },
    },
}
