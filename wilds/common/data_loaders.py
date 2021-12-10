import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler
from wilds.common.utils import get_counts, split_into_groups

def get_train_loader(loader, dataset, batch_size,
        uniform_over_groups=None, grouper=None, distinct_groups=True, n_groups_per_batch=None, **loader_kwargs):
    """
    Constructs and returns the data loader for training.
    Args:
        - loader (str): Loader type. 'standard' for standard loaders and 'group' for group loaders,
                        which first samples groups and then samples a fixed number of examples belonging
                        to each group.
        - dataset (WILDSDataset or WILDSSubset): Data
        - batch_size (int): Batch size
        - uniform_over_groups (None or bool): Whether to sample the groups uniformly or according
                                              to the natural data distribution.
                                              Setting to None applies the defaults for each type of loaders.
                                              For standard loaders, the default is False. For group loaders,
                                              the default is True.
        - grouper (Grouper): Grouper used for group loaders or for uniform_over_groups=True
        - distinct_groups (bool): Whether to sample distinct_groups within each minibatch for group loaders.
        - n_groups_per_batch (int): Number of groups to sample in each minibatch for group loaders.
        - loader_kwargs: kwargs passed into torch DataLoader initialization.
    Output:
        - data loader (DataLoader): Data loader.
    """
    if loader == 'standard':
        if uniform_over_groups is None or not uniform_over_groups:
            return DataLoader(
                dataset,
                shuffle=True, # Shuffle training dataset
                sampler=None,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs)
        else:
            assert grouper is not None
            groups, group_counts = grouper.metadata_to_group(
                dataset.metadata_array,
                return_counts=True)
            group_weights = 1 / group_counts
            weights = group_weights[groups]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
            return DataLoader(
                dataset,
                shuffle=False, # The WeightedRandomSampler already shuffles
                sampler=sampler,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs)

    elif loader == 'group':
        if uniform_over_groups is None:
            uniform_over_groups = True
        assert grouper is not None
        assert n_groups_per_batch is not None
        if n_groups_per_batch > grouper.n_groups:
            raise ValueError(f'n_groups_per_batch was set to {n_groups_per_batch} but there are only {grouper.n_groups} groups specified.')

        group_ids = grouper.metadata_to_group(dataset.metadata_array)
        batch_sampler = GroupSampler(
            group_ids=group_ids,
            batch_size=batch_size,
            n_groups_per_batch=n_groups_per_batch,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups)

        return DataLoader(dataset,
              shuffle=None,
              sampler=None,
              collate_fn=dataset.collate,
              batch_sampler=batch_sampler,
              drop_last=False,
              **loader_kwargs)

def get_eval_loader(loader, dataset, batch_size, grouper=None, **loader_kwargs):
    """
    Constructs and returns the data loader for evaluation.
    Args:
        - loader (str): Loader type. 'standard' for standard loaders.
        - dataset (WILDSDataset or WILDSSubset): Data
        - batch_size (int): Batch size
        - loader_kwargs: kwargs passed into torch DataLoader initialization.
    Output:
        - data loader (DataLoader): Data loader.
    """
    if loader == 'standard':
        return DataLoader(
            dataset,
            shuffle=False, # Do not shuffle eval datasets
            sampler=None,
            collate_fn=dataset.collate,
            batch_size=batch_size,
            **loader_kwargs)

class GroupSampler:
    """
        Constructs batches by first sampling groups,
        then sampling data from those groups.
        It drops the last batch if it's incomplete.
    """
    def __init__(self, group_ids, batch_size, n_groups_per_batch,
                 uniform_over_groups, distinct_groups):

        if batch_size % n_groups_per_batch != 0:
            raise ValueError(f'batch_size ({batch_size}) must be evenly divisible by n_groups_per_batch ({n_groups_per_batch}).')
        if len(group_ids) < batch_size:
            raise ValueError(f'The dataset has only {len(group_ids)} examples but the batch size is {batch_size}. There must be enough examples to form at least one complete batch.')

        self.group_ids = group_ids
        self.unique_groups, self.group_indices, unique_counts = split_into_groups(group_ids)

        self.distinct_groups = distinct_groups
        self.n_groups_per_batch = n_groups_per_batch
        self.n_points_per_group = batch_size // n_groups_per_batch

        self.dataset_size = len(group_ids)
        self.num_batches = self.dataset_size // batch_size

        if uniform_over_groups: # Sample uniformly over groups
            self.group_prob = None
        else: # Sample a group proportionately to its size
            self.group_prob = unique_counts.numpy() / unique_counts.numpy().sum()

    def __iter__(self):
        for batch_id in range(self.num_batches):
            # Note that we are selecting group indices rather than groups
            groups_for_batch = np.random.choice(
                len(self.unique_groups),
                size=self.n_groups_per_batch,
                replace=(not self.distinct_groups),
                p=self.group_prob)
            sampled_ids = [
                np.random.choice(
                    self.group_indices[group],
                    size=self.n_points_per_group,
                    replace=len(self.group_indices[group]) <= self.n_points_per_group, # False if the group is larger than the sample size
                    p=None)
                for group in groups_for_batch]

            # Flatten
            sampled_ids = np.concatenate(sampled_ids)
            yield sampled_ids

    def __len__(self):
        return self.num_batches
