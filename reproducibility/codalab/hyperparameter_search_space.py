HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {
            "lr": [1e-6, 2e-6, 1e-5, 2e-5],
            "weight_decay": [0.01],
        },
        "civilcomments": {
            "lr": [1e-6, 2e-6, 1e-5, 2e-5],
        },
        "camelyon17": {
            "lr": [1e-4, 1e-3, 1e-2],
            "weight_decay": [0, 1e-3, 1e-2],
        },
        "iwildcam": {
            "lr": [1e-5, 3e-5, 1e-4],
            "weight_decay": [0, 1e-3, 1e-2],
        },
        "fmow": {
            "lr": [1e-2, 1e-3, 1e-4, 1e-5],
            "weight_decay": [0, 1e-0, 1e-1, 1e-2, 1e-3],
        },
        "poverty": {
            "lr": [1e-2, 1e-3, 1e-4, 1e-5],
            "weight_decay": [0, 1e-0, 1e-1, 1e-2, 1e-3],
        },
        "py150": {
            "lr": [8e-4, 8e-5, 8e-6],
            "weight_decay": [0, 0.01, 0.1],
        },
    },
    "algorithms": {
        "irm": {"irm_lambda": [1, 10, 100, 1000]},
        "deepCORAL": {"coral_penalty_weight": [0.1, 1, 10]},
    },
}
