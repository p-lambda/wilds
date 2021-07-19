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
