import sys
import os
import csv
import argparse
import random
from pathlib import Path
import numpy as np
import torch

try:
    import wandb
except Exception as e:
    pass

def update_average(prev_avg, prev_counts, curr_avg, curr_counts):
    denom = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denom += (denom==0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denom==0:
            return 0.
    else:
        raise ValueError('Type of curr_counts not recognized')
    prev_weight = prev_counts/denom
    curr_weight = curr_counts/denom
    return prev_weight*prev_avg + curr_weight*curr_avg

# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save(algorithm, epoch, best_val_metric, path):
    state = {}
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)

def load(algorithm, path):
    state = torch.load(path)
    algorithm.load_state_dict(state['algorithm'])
    return state['epoch'], state['best_val_metric']

def log_group_data(datasets, grouper, logger):
    for k, dataset in datasets.items():
        name = dataset['name']
        dataset = dataset['dataset']
        logger.write(f'{name} data...\n')
        if grouper is None:
            logger.write(f'    n = {len(dataset)}\n')
        else:
            _, group_counts = grouper.metadata_to_group(
                dataset.metadata_array,
                return_counts=True)
            group_counts = group_counts.tolist()
            for group_idx in range(grouper.n_groups):
                logger.write(f'    {grouper.group_str(group_idx)}: n = {group_counts[group_idx]:.0f}\n')
    logger.flush()

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class BatchLogger:
    def __init__(self, csv_path, mode='w', use_wandb=False):
        self.path = csv_path
        self.mode = mode
        self.file = open(csv_path, mode)
        self.is_initialized = False

        # Use Weights and Biases for logging
        self.use_wandb = use_wandb
        if use_wandb:
            self.split = Path(csv_path).stem

    def setup(self, log_dict):
        columns = log_dict.keys()
        # Move epoch and batch to the front if in the log_dict
        for key in ['batch', 'epoch']:
            if key in columns:
                columns = [key] + [k for k in columns if k != key]

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if self.mode=='w' or (not os.path.exists(self.path)) or os.path.getsize(self.path)==0:
            self.writer.writeheader()
        self.is_initialized = True

    def log(self, log_dict):
        if self.is_initialized is False:
            self.setup(log_dict)
        self.writer.writerow(log_dict)
        self.flush()

        if self.use_wandb:
            results = {}
            for key in log_dict:
                new_key = f'{self.split}/{key}'
                results[new_key] = log_dict[key]
            wandb.log(results)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_config(config, logger):
    for name, val in vars(config).items():
        logger.write(f'{name.replace("_"," ").capitalize()}: {val}\n')
    logger.write('\n')

def initialize_wandb(config):
    name = config.dataset + '_' + config.algorithm + '_' + config.log_dir
    wandb.init(name=name,
               project=f"wilds")
    wandb.config.update(config)
