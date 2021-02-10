ID_HYPERPARAMETER_SEARCH_SPACE = {
    # Skip OGB and CivilComments as ID validation sets do not exist for these datasets
    "datasets": {
        "amazon": {"lr": [1e-6, 2e-6, 1e-5, 2e-5]},
        "camelyon17": {
            "lr": [1e-4, 1e-3, 1e-2],
            "weight_decay": [0, 1e-3, 1e-2],
        },
        "iwildcam": {
            "lr": [1e-3, 1e-4, 1e-5],
            "weight_decay": [0, 1e-4, 1e-5],
        },
        "fmow": {
            "lr": [1e-2, 1e-3, 1e-4, 1e-5],
            "weight_decay": [0, 1e-0, 1e-1, 1e-2, 1e-3],
        },
        "poverty": {
            "lr": [1e-2, 1e-3, 1e-4, 1e-5],
            "weight_decay": [0, 1e-0, 1e-1, 1e-2, 1e-3],
        },
    }
}
