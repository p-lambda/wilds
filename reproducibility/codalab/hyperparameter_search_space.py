MAX_BATCH_SIZES = {
    "amazon": 32,
    "civilcomments": 64,
    "camelyon17": 224,
    "iwildcam": 32,
    "fmow": 96,
    "poverty": 160,
}

ERM_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "batch_size": [MAX_BATCH_SIZES["amazon"]],
            "lr": [-6, -4],
            "weight_decay": [-3, -1],
        },
        "civilcomments": {
            "batch_size": [MAX_BATCH_SIZES["civilcomments"]],
            "lr": [-6, -4],
        },
        "camelyon17": {
            "batch_size": [MAX_BATCH_SIZES["camelyon17"]],
            "lr": [-4, -1],
            "weight_decay": [-4, -1],
        },
        "iwildcam": {
            "batch_size": [MAX_BATCH_SIZES["iwildcam"]],
            "lr": [-5, -3],
            "weight_decay": [-4, -1],
        },
        "fmow": {
            "batch_size": [MAX_BATCH_SIZES["fmow"]],
            "lr": [-5, -1],
            "weight_decay": [-5, 0],
        },
        "poverty": {
            "batch_size": [MAX_BATCH_SIZES["poverty"]],
            "lr": [-5, -1],
            "weight_decay": [-4, 0],
        },
    },
}

ERM_AUGMENT_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "batch_size": [MAX_BATCH_SIZES["amazon"]],
            "lr": [-6, -4],
            "weight_decay": [-3, -1],
            "additional_train_transform": ["randaugment"],
        },
        "civilcomments": {
            "batch_size": [MAX_BATCH_SIZES["civilcomments"]],
            "lr": [-6, -4],
            "additional_train_transform": ["randaugment"],
        },
        "camelyon17": {
            "batch_size": [MAX_BATCH_SIZES["camelyon17"]],
            "lr": [-4, -1],
            "weight_decay": [-4, -1],
            "additional_train_transform": ["randaugment"],
        },
        "iwildcam": {
            "batch_size": [MAX_BATCH_SIZES["iwildcam"]],
            "lr": [-5, -3],
            "weight_decay": [-4, -1],
            "additional_train_transform": ["randaugment"],
        },
        "fmow": {
            "batch_size": [MAX_BATCH_SIZES["fmow"]],
            "lr": [-5, -1],
            "weight_decay": [-5, 0],
            "additional_train_transform": ["randaugment"],
        },
        "poverty": {
            "batch_size": [MAX_BATCH_SIZES["poverty"]],
            "lr": [-5, -1],
            "weight_decay": [-4, 0],
            "additional_train_transform": ["randaugment"],
        },
    },
}

CORAL_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "lr": [-6, -4],
            "weight_decay": [-3, -1],
            "coral_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "civilcomments": {
            "lr": [-6, -4],
            "coral_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "camelyon17": {
            "lr": [-4, -1],
            "weight_decay": [-4, -1],
            "coral_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "iwildcam": {
            "lr": [-5, -3],
            "weight_decay": [-4, -1],
            "coral_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "fmow": {
            "lr": [-5, -1],
            "weight_decay": [-5, 0],
            "coral_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "poverty": {
            "lr": [-5, -1],
            "weight_decay": [-4, 0],
            "coral_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
    },
}

DANN_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "weight_decay": [-3, -1],
            "dann_classifier_lr": [-6, -4],
            "dann_discriminator_lr": [-6, -4],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "civilcomments": {
            "dann_classifier_lr": [-6, -4],
            "dann_discriminator_lr": [-6, -4],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "camelyon17": {
            "weight_decay": [-4, -1],
            "dann_classifier_lr": [-4, -1],
            "dann_discriminator_lr": [-4, -1],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "iwildcam": {
            "weight_decay": [-4, -1],
            "dann_classifier_lr": [-5, -3],
            "dann_discriminator_lr": [-5, -3],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "fmow": {
            "weight_decay": [-5, 0],
            "dann_classifier_lr": [-5, -1],
            "dann_discriminator_lr": [-5, -1],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "poverty": {
            "weight_decay": [-4, 0],
            "dann_classifier_lr": [-5, -1],
            "dann_discriminator_lr": [-5, -1],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
    },
}

FIXMATCH_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "weight_decay": [-3, -1],
            "dann_classifier_lr": [-6, -4],
            "dann_discriminator_lr": [-6, -4],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "civilcomments": {
            "dann_classifier_lr": [-6, -4],
            "dann_discriminator_lr": [-6, -4],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "camelyon17": {
            "weight_decay": [-4, -1],
            "dann_classifier_lr": [-4, -1],
            "dann_discriminator_lr": [-4, -1],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "iwildcam": {
            "weight_decay": [-4, -1],
            "dann_classifier_lr": [-5, -3],
            "dann_discriminator_lr": [-5, -3],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "fmow": {
            "weight_decay": [-5, 0],
            "lr": [-5, -1],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
        "poverty": {
            "weight_decay": [-4, 0],
            "dann_classifier_lr": [-5, -1],
            "dann_discriminator_lr": [-5, -1],
            "dann_penalty_weight": [-2, 2],
            "unlabeled_batch_size_frac": [0.5, 0.9],
        },
    },
}