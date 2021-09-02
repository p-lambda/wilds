import numpy as np
from wilds.common.utils import avg_over_groups, get_counts, numel
import torch

class Metric:
    """
    Parent class for metrics.
    """
    def __init__(self, name):
        self._name = name

    def _compute(self, y_pred, y_true):
        """
        Helper function for computing the metric.
        Subclasses should implement this.
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - metric (0-dim tensor): metric
        """
        return NotImplementedError

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (0-dim tensor): Worst-case metric
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Metric name.
        Used to name the key in the results dictionaries returned by the metric.
        """
        return self._name

    @property
    def agg_metric_field(self):
        """
        The name of the key in the results dictionary returned by Metric.compute().
        This should correspond to the aggregate metric computed on all of y_pred and y_true,
        in contrast to a group-wise evaluation.
        """
        return f'{self.name}_all'

    def group_metric_field(self, group_idx):
        """
        The name of the keys corresponding to individual group evaluations
        in the results dictionary returned by Metric.compute_group_wise().
        """
        return f'{self.name}_group:{group_idx}'

    @property
    def worst_group_metric_field(self):
        """
        The name of the keys corresponding to the worst-group metric
        in the results dictionary returned by Metric.compute_group_wise().
        """
        return f'{self.name}_wg'

    def group_count_field(self, group_idx):
        """
        The name of the keys corresponding to each group's count
        in the results dictionary returned by Metric.compute_group_wise().
        """
        return f'count_group:{group_idx}'

    def compute(self, y_pred, y_true, return_dict=True):
        """
        Computes metric. This is a wrapper around _compute.
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - metric (0-dim tensor): metric. If the inputs are empty, returns tensor(0.)
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.agg_metric_field to avg_metric
        """
        if numel(y_true) == 0:
            agg_metric = torch.tensor(0., device=y_true.device)
        else:
            agg_metric = self._compute(y_pred, y_true)
        if return_dict:
            results = {
                self.agg_metric_field: agg_metric.item()
            }
            return results
        else:
            return agg_metric

    def compute_group_wise(self, y_pred, y_true, g, n_groups, return_dict=True):
        """
        Computes metrics for each group. This is a wrapper around _compute.
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
            - g (Tensor): groups
            - n_groups (int): number of groups
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - group_metrics (Tensor): tensor of size (n_groups, ) including the average metric for each group
            - group_counts (Tensor): tensor of size (n_groups, ) including the group count
            - worst_group_metric (0-dim tensor): worst-group metric
            - For empty inputs/groups, corresponding metrics are tensor(0.)
        Output (return_dict=True):
            - results (dict): Dictionary of results
        """
        group_metrics, group_counts, worst_group_metric = self._compute_group_wise(y_pred, y_true, g, n_groups)
        if return_dict:
            results = {}
            for group_idx in range(n_groups):
                results[self.group_metric_field(group_idx)] = group_metrics[group_idx].item()
                results[self.group_count_field(group_idx)] = group_counts[group_idx].item()
            results[self.worst_group_metric_field] = worst_group_metric.item()
            return results
        else:
            return group_metrics, group_counts, worst_group_metric

    def _compute_group_wise(self, y_pred, y_true, g, n_groups):
        group_metrics = []
        group_counts = get_counts(g, n_groups)
        for group_idx in range(n_groups):
            if group_counts[group_idx]==0:
                group_metrics.append(torch.tensor(0., device=g.device))
            else:
                group_metrics.append(
                    self._compute(
                        y_pred[g == group_idx],
                        y_true[g == group_idx]))

        group_metrics = torch.stack(group_metrics)
        worst_group_metric = self.worst(group_metrics[group_counts>0])

        return group_metrics, group_counts, worst_group_metric

class ElementwiseMetric(Metric):
    """
    Averages.
    """
    def _compute_element_wise(self, y_pred, y_true):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        raise NotImplementedError

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (0-dim tensor): Worst-case metric
        """
        raise NotImplementedError

    def _compute(self, y_pred, y_true):
        """
        Helper function for computing the metric.
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - avg_metric (0-dim tensor): average of element-wise metrics
        """
        element_wise_metrics = self._compute_element_wise(y_pred, y_true)
        avg_metric = element_wise_metrics.mean()
        return avg_metric

    def _compute_group_wise(self, y_pred, y_true, g, n_groups):
        element_wise_metrics = self._compute_element_wise(y_pred, y_true)
        group_metrics, group_counts = avg_over_groups(element_wise_metrics, g, n_groups)
        worst_group_metric = self.worst(group_metrics[group_counts>0])
        return group_metrics, group_counts, worst_group_metric

    @property
    def agg_metric_field(self):
        """
        The name of the key in the results dictionary returned by Metric.compute().
        """
        return f'{self.name}_avg'

    def compute_element_wise(self, y_pred, y_true, return_dict=True):
        """
        Computes element-wise metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.name to element_wise_metrics
        """
        element_wise_metrics = self._compute_element_wise(y_pred, y_true)
        batch_size = y_pred.size()[0]
        assert element_wise_metrics.dim()==1 and element_wise_metrics.numel()==batch_size

        if return_dict:
            return {self.name: element_wise_metrics}
        else:
            return element_wise_metrics

    def compute_flattened(self, y_pred, y_true, return_dict=True):
        flattened_metrics = self.compute_element_wise(y_pred, y_true, return_dict=False)
        index = torch.arange(y_true.numel())
        if return_dict:
            return {self.name: flattened_metrics, 'index': index}
        else:
            return flattened_metrics, index

class MultiTaskMetric(Metric):
    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        raise NotImplementedError

    def _compute(self, y_pred, y_true):
        flattened_metrics, _ = self.compute_flattened(y_pred, y_true, return_dict=False)
        if flattened_metrics.numel()==0:
            return torch.tensor(0., device=y_true.device)
        else:
            return flattened_metrics.mean()

    def _compute_group_wise(self, y_pred, y_true, g, n_groups):
        flattened_metrics, indices = self.compute_flattened(y_pred, y_true, return_dict=False)
        flattened_g = g[indices]
        group_metrics, group_counts = avg_over_groups(flattened_metrics, flattened_g, n_groups)
        worst_group_metric = self.worst(group_metrics[group_counts>0])
        return group_metrics, group_counts, worst_group_metric

    def compute_flattened(self, y_pred, y_true, return_dict=True):
        is_labeled = ~torch.isnan(y_true)
        batch_idx = torch.where(is_labeled)[0]
        flattened_y_pred = y_pred[is_labeled]
        flattened_y_true = y_true[is_labeled]
        flattened_metrics = self._compute_flattened(flattened_y_pred, flattened_y_true)
        if return_dict:
            return {self.name: flattened_metrics, 'index': batch_idx}
        else:
            return flattened_metrics, batch_idx
