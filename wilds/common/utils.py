import torch, torch_scatter
import numpy as np
from torch.utils.data import Subset
from pandas.api.types import CategoricalDtype

def minimum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel()==0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].min()
    elif isinstance(numbers, np.ndarray):
        if numbers.size==0:
            return np.array(empty_val)
        else:
            return np.nanmin(numbers)
    else:
        if len(numbers)==0:
            return empty_val
        else:
            return min(numbers)

def maximum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel()==0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].max()
    elif isinstance(numbers, np.ndarray):
        if numbers.size==0:
            return np.array(empty_val)
        else:
            return np.nanmax(numbers)
    else:
        if len(numbers)==0:
            return empty_val
        else:
            return max(numbers)

def split_into_groups(g):
    """
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts

def get_counts(g, n_groups):
    """
    This differs from split_into_groups in how it handles missing groups.
    get_counts always returns a count Tensor of length n_groups,
    whereas split_into_groups returns a unique_counts Tensor
    whose length is the number of unique groups present in g.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - counts (Tensor): A list of length n_groups, denoting the count of each group.
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    counts = torch.zeros(n_groups, device=g.device)
    counts[unique_groups] = unique_counts.float()
    return counts

def avg_over_groups(v, g, n_groups):
    """
    Args:
        v (Tensor): Vector containing the quantity to average over.
        g (Tensor): Vector of the same length as v, containing group information.
    Returns:
        group_avgs (Tensor): Vector of length num_groups
        group_counts (Tensor)
    """
    assert v.device==g.device
    device = v.device
    assert v.numel()==g.numel()
    group_count = get_counts(g, n_groups)
    group_avgs = torch_scatter.scatter(src=v, index=g, dim_size=n_groups, reduce='mean')
    return group_avgs, group_count

def map_to_id_array(df, ordered_map={}):
    maps = {}
    array = np.zeros(df.shape)
    for i, c in enumerate(df.columns):
        if c in ordered_map:
            category_type = CategoricalDtype(categories=ordered_map[c], ordered=True)
        else:
            category_type = 'category'
        series = df[c].astype(category_type)
        maps[c] = series.cat.categories.values
        array[:,i] = series.cat.codes.values
    return maps, array

def subsample_idxs(idxs, num=5000, take_rest=False, seed=None):
    seed = (seed + 541433) if seed is not None else None
    rng = np.random.default_rng(seed)

    idxs = idxs.copy()
    rng.shuffle(idxs)
    if take_rest:
        idxs = idxs[num:]
    else:
        idxs = idxs[:num]
    return idxs


def shuffle_arr(arr, seed=None):
    seed = (seed + 548207) if seed is not None else None
    rng = np.random.default_rng(seed)

    arr = arr.copy()
    rng.shuffle(arr)
    return arr

def threshold_at_recall(y_pred, y_true, global_recall=60):
    """ Calculate the model threshold to use to achieve a desired global_recall level. Assumes that
    y_true is a vector of the true binary labels."""
    return np.percentile(y_pred[y_true == 1], 100-global_recall)
