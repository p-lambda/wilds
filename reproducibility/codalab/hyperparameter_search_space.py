ID_HYPERPARAMETER_SEARCH_SPACE = {
    "datasets": {
        "amazon": {"lr": [1e-6, 2e-6, 1e-5, 2e-5], "n_epochs": [1, 2, 3]},
        "camelyon17": {
            "lr": [1e-4, 1e-3, 1e-2],
            "weight_decay": [0, 1e-3, 1e-2],
        },
        "iwildcam": {
            "lr": [1e-3, 1e-4, 1e-5],
            "weight_decay": [0, 1e-4, 1e-5],
        },
    }
}
