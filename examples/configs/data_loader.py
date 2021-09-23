loader_defaults = {
    'loader_kwargs': {
        'num_workers': 4,
        'pin_memory': True,
    },
    'unlabeled_loader_kwargs': {
        'num_workers': 8,
        'pin_memory': True,
    },
    'n_groups_per_batch': 4,
}
