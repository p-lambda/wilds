CORAL_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "lr": [-6, -4],
            "weight_decay": [-3, -1],
            "coral_penalty_weight": [-2, 2],
        },
        "civilcomments": {
            "lr": [-6, -4],
            "coral_penalty_weight": [-2, 2],
        },
        "camelyon17": {
            "lr": [-4, -1],
            "weight_decay": [-4, -1],
            "coral_penalty_weight": [-2, 2],
        },
        "iwildcam": {
            "lr": [-5, -3],
            "weight_decay": [-4, -1],
            "coral_penalty_weight": [-2, 2],
        },
        "fmow": {
            "lr": [-5, -1],
            "weight_decay": [-5, 0],
            "coral_penalty_weight": [-2, 2],
        },
        "poverty": {
            "lr": [-5, -1],
            "weight_decay": [-4, 0],
            "coral_penalty_weight": [-2, 2],
        },
        "py150": {
            "lr": [-7, -4],
            "weight_decay": [-3, -1],
            "coral_penalty_weight": [-2, 2],
        },
    },
}

DANN_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "lr": [-6, -4],
            "weight_decay": [-3, -1],
            "dann_classifier_lr": [-1, 0],
            "dann_discriminator_lr": [1],
            "dann_penalty_weight": [-2, 2],
        },
        "civilcomments": {
            "lr": [-6, -4],
            "dann_classifier_lr": [-1, 0],
            "dann_discriminator_lr": [1],
            "dann_penalty_weight": [-2, 2],
        },
        "camelyon17": {
            "lr": [-4, -1],
            "weight_decay": [-4, -1],
            "dann_classifier_lr": [-1, 0],
            "dann_discriminator_lr": [1],
            "dann_penalty_weight": [-2, 2],
        },
        "iwildcam": {
            "lr": [-5, -3],
            "weight_decay": [-4, -1],
            "dann_classifier_lr": [-1, 0],
            "dann_discriminator_lr": [1],
            "dann_penalty_weight": [-2, 2],
        },
        # "fmow": {
        #     "lr": [-5, -1],
        #     "weight_decay": [-5, 0],
        #     "dann_classifier_lr": [-1, 0],
        #     "dann_discriminator_lr": [1],
        #     "dann_penalty_weight": [-2, 2],
        # },
        "fmow": {
            "lr": [-5, -1],
            "weight_decay": [-5, 0],
            "dann_classifier_lr": [-4, -2],
            "dann_discriminator_lr": [1],
            "dann_penalty_weight": [-2, 2],
        },
        "poverty": {
            "lr": [-5, -1],
            "weight_decay": [-4, 0],
            "dann_classifier_lr": [-1, 0],
            "dann_discriminator_lr": [1],
            "dann_penalty_weight": [-2, 2],
        },
        "py150": {
            "lr": [-7, -4],
            "weight_decay": [-3, -1],
            "dann_classifier_lr": [-1, 0],
            "dann_discriminator_lr": [1],
            "dann_penalty_weight": [-2, 2],
        },
    },
}
