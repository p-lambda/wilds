from configs.algorithm import algorithm_defaults
from configs.model import model_defaults
from configs.scheduler import scheduler_defaults
from configs.data_loader import loader_defaults
from configs.datasets import dataset_defaults, split_defaults

def populate_defaults(config):
    """Populates hyperparameters with defaults implied by choices
    of other hyperparameters."""
    assert config.dataset is not None, 'dataset must be specified'
    assert config.algorithm is not None, 'algorithm must be specified'
    # implied defaults from choice of dataset
    config = populate_config(
        config, 
        dataset_defaults[config.dataset]
    )

    # implied defaults from choice of split
    if config.dataset in split_defaults and config.split_scheme in split_defaults[config.dataset]:
        config = populate_config(
            config, 
            split_defaults[config.dataset][config.split_scheme]
        )
    
    # implied defaults from choice of algorithm
    config = populate_config(
        config, 
        algorithm_defaults[config.algorithm]
    )

    # implied defaults from choice of loader
    config = populate_config(
        config, 
        loader_defaults
    )
    # implied defaults from choice of model
    if config.model: config = populate_config(
        config, 
        model_defaults[config.model],
    )
    
    # implied defaults from choice of scheduler
    if config.scheduler: config = populate_config(
        config, 
        scheduler_defaults[config.scheduler]
    )

    # misc implied defaults
    if config.groupby_fields is None:
        config.no_group_logging = True
    config.no_group_logging = bool(config.no_group_logging)

    # basic checks
    required_fields = [
        'split_scheme', 'train_loader', 'uniform_over_groups', 'batch_size', 'eval_loader', 'model', 'loss_function', 
        'val_metric', 'val_metric_decreasing', 'n_epochs', 'optimizer', 'lr', 'weight_decay',
        ] 
    for field in required_fields:
        assert getattr(config, field) is not None, f"Must manually specify {field} for this setup."

    return config

def populate_config(config, template: dict, force_compatibility=False):
    """Populates missing (key, val) pairs in config with (key, val) in template.
    Example usage: populate config with defaults
    Args:
        - config: namespace
        - template: dict
        - force_compatibility: option to raise errors if config.key != template[key]
    """
    if template is None:
        return config

    d_config = vars(config)
    for key, val in template.items():
        if not isinstance(val, dict): # config[key] expected to be a non-index-able
            if key not in d_config or d_config[key] is None:
                d_config[key] = val
            elif d_config[key] != val and force_compatibility:
                raise ValueError(f"Argument {key} must be set to {val}")
                
        else: # config[key] expected to be a kwarg dict
            for kwargs_key, kwargs_val in val.items():
                if kwargs_key not in d_config[key] or d_config[key][kwargs_key] is None:
                    d_config[key][kwargs_key] = kwargs_val
                elif d_config[key][kwargs_key] != kwargs_val and force_compatibility:
                    raise ValueError(f"Argument {key}[{kwargs_key}] must be set to {val}")
    return config
