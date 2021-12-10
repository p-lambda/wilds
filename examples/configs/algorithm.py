algorithm_defaults = {
    'ERM': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'randaugment_n': 2,     # When running ERM + data augmentation
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
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
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
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        # 'dann_penalty_weight': 0, # TODO: fill in
        # 'dann_classifier_lr': 0, # TODO: fill in
        # 'dann_featurizer_lr': 0, # TODO: fill in
        # 'dann_discriminator_lr': 0, # TODO: fill in
    },
    'AFN': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'use_hafn': False,
        'afn_penalty_weight': 0.01,
        'safn_delta_r': 1.0,
        'hafn_r': 1.0,
        'additional_train_transform': 'randaugment',    # Apply strong augmentation to labeled & unlabeled examples
        'randaugment_n': 2,
        # 'afn_penalty_weight': 0, #TODO: fill in
        # 'safn_delta_r': 0, #TODO: fill in
        # 'hafn_r': 0, #TODO: fill in
        # 'use_hafn': False, #TODO: fill in
    },
    'FixMatch': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_lambda': 1, # TODO: fill in
        'self_training_threshold': 0.7, # TODO: fill in
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled examples
    },
    'PseudoLabel': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_lambda': 1, # TODO: fill in
        'self_training_threshold': 0.7, # TODO: fill in
        'pseudolabel_T2': 0.4, # TODO: fill in
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'NoisyStudent': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'noisystudent_add_dropout': True,
        'noisystudent_dropout_rate': 0.5,
        'noisystudent_soft_pseudolabels': False,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    }
}
