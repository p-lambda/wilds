algorithm_defaults = {
    'ERM': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'randaugment_n': 2,     # If we are running ERM + data augmentation
    },
    'groupDRO': {
        'train_loader': 'standard',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'group_dro_step_size': 0.01,
    },
    'deepCORAL': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'coral_penalty_weight': 1.,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled examples
    },
    'IRM': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'irm_lambda': 100.,
        'irm_penalty_anneal_iters': 500,
    },
    'DANN': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled examples
    },
    'FixMatch': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_lambda': 1,
        'self_training_threshold': 0.7,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled examples
    },
    'PseudoLabel': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_lambda': 1,
        'self_training_threshold': 0.7,
        'pseudolabel_T2': 0.4,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled examples
    },
    'NoisyStudent': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'dropout_rate': 0.5,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled examples
    }
}
