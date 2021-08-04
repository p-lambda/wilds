import torch, time
import numpy as np
from algorithms.algorithm import Algorithm
from utils import update_average
from scheduler import step_scheduler
from wilds.common.utils import get_counts

class GroupAlgorithm(Algorithm):
    """
    Parent class for algorithms with group-wise logging.
    Also handles schedulers.
    """
    def __init__(self, device, grouper, logged_metrics, logged_fields, schedulers, scheduler_metric_names, no_group_logging, **kwargs):
        """
        Args:
            - device: torch device
            - grouper (Grouper): defines groups for which we compute/log stats for
            - logged_metrics (list of Metric):
            - logged_fields (list of str):
        """
        super().__init__(device)
        self.grouper = grouper
        self.group_prefix = 'group_'
        self.count_field = 'count'
        self.group_count_field = f'{self.group_prefix}{self.count_field}'

        self.logged_metrics = logged_metrics
        self.logged_fields = logged_fields

        self.schedulers = schedulers
        self.scheduler_metric_names = scheduler_metric_names
        self.no_group_logging = no_group_logging

    def update_log(self, results):
        """
        Updates the internal log, Algorithm.log_dict
        Args:
            - results (dictionary)
        """
        results = self.sanitize_dict(results, to_out_device=False)
        # check all the fields exist
        for field in self.logged_fields:
            assert field in results, f"field {field} missing"
        # compute statistics for the current batch
        batch_log = {}
        with torch.no_grad():
            for m in self.logged_metrics:
                if not self.no_group_logging:
                    group_metrics, group_counts, worst_group_metric = m.compute_group_wise(
                        results['y_pred'],
                        results['y_true'],
                        results['g'],
                        self.grouper.n_groups,
                        return_dict=False)
                    batch_log[f'{self.group_prefix}{m.name}'] = group_metrics
                batch_log[m.agg_metric_field] = m.compute(
                    results['y_pred'],
                    results['y_true'],
                    return_dict=False).item()
            count = results['y_true'].numel()

        # transfer other statistics in the results dictionary
        for field in self.logged_fields:
            if field.startswith(self.group_prefix) and self.no_group_logging:
                continue
            v = results[field]
            if isinstance(v, torch.Tensor) and v.numel()==1:
                batch_log[field] = v.item()
            else:
                if isinstance(v, torch.Tensor):
                    assert v.numel()==self.grouper.n_groups, "Current implementation deals only with group-wise statistics or a single-number statistic"
                    assert field.startswith(self.group_prefix)
                batch_log[field] = v

        # update the log dict with the current batch
        if not self._has_log: # since it is the first log entry, just save the current log
            self.log_dict = batch_log
            if not self.no_group_logging:
                self.log_dict[self.group_count_field] = group_counts
            self.log_dict[self.count_field] = count
        else: # take a running average across batches otherwise
            for k, v in batch_log.items():
                if k.startswith(self.group_prefix):
                    if self.no_group_logging:
                        continue
                    self.log_dict[k] = update_average(self.log_dict[k], self.log_dict[self.group_count_field], v, group_counts)
                else:
                    self.log_dict[k] = update_average(self.log_dict[k], self.log_dict[self.count_field], v, count)
            if not self.no_group_logging:
                self.log_dict[self.group_count_field] += group_counts
            self.log_dict[self.count_field] += count
        self._has_log = True

    def get_log(self):
        """
        Sanitizes the internal log (Algorithm.log_dict) and outputs it.
        """
        sanitized_log = {}
        for k, v in self.log_dict.items():
            if k.startswith(self.group_prefix):
                field = k[len(self.group_prefix):]
                for g in range(self.grouper.n_groups):
                    # set relevant values to NaN depending on the group count
                    count = self.log_dict[self.group_count_field][g].item()
                    if count==0 and k!=self.group_count_field:
                        outval = np.nan
                    else:
                        outval = v[g].item()
                    # add to dictionary with an appropriate name
                    # in practice, it is saving each value as {field}_group:{g}
                    added = False
                    for m in self.logged_metrics:
                        if field==m.name:
                            sanitized_log[m.group_metric_field(g)] = outval
                            added = True
                    if k==self.group_count_field:
                        sanitized_log[self.loss.group_count_field(g)] = outval
                        added = True
                    elif not added:
                        sanitized_log[f'{field}_group:{g}'] = outval
            else:
                assert not isinstance(v, torch.Tensor)
                sanitized_log[k] = v
        return sanitized_log

    def step_schedulers(self, is_epoch, metrics={}, log_access=False):
        """
        Updates the scheduler after an epoch.
        If a scheduler is updated based on a metric (SingleModelAlgorithm.scheduler_metric),
        then it first looks for an entry in metrics_dict and then in its internal log
        (SingleModelAlgorithm.log_dict) if log_access is True.
        Args:
            - metrics_dict (dictionary)
            - log_access (bool): whether the scheduler_metric can be fetched from internal log
                                 (self.log_dict)
        """
        for scheduler, metric_name in zip(self.schedulers, self.scheduler_metric_names):
            if scheduler is None:
                continue
            if is_epoch and scheduler.step_every_batch:
                continue
            if (not is_epoch) and (not scheduler.step_every_batch):
                continue
            self._step_specific_scheduler(
                scheduler=scheduler,
                metric_name=metric_name,
                metrics=metrics,
                log_access=log_access)

    def _step_specific_scheduler(self, scheduler, metric_name, metrics, log_access):
        """
        Helper function for updating scheduler
        Args:
            - scheduler: scheduler to update
            - is_epoch (bool): epoch-wise update if set to True, batch-wise update otherwise
            - metric_name (str): name of the metric (key in metrics or log dictionary) to use for updates
            - metrics (dict): a dictionary of metrics that can beused for scheduler updates
            - log_access (bool): whether metrics from self.get_log() can be used to update schedulers
        """
        if not scheduler.use_metric or metric_name is None:
            metric = None
        elif metric_name in metrics:
            metric = metrics[metric_name]
        elif log_access:
            sanitized_log_dict = self.get_log()
            if metric_name in sanitized_log_dict:
                metric = sanitized_log_dict[metric_name]
            else:
                raise ValueError('scheduler metric not recognized')
        else:
            raise ValueError('scheduler metric not recognized')
        step_scheduler(scheduler, metric)

    def get_pretty_log_str(self):
        """
        Output:
            - results_str (str)
        """
        results_str = ''

        # Get sanitized log dict
        log = self.get_log()

        # Process aggregate logged fields
        for field in self.logged_fields:
            if field.startswith(self.group_prefix):
                continue
            results_str += (
                f'{field}: {log[field]:.3f}\n'
            )

        # Process aggregate logged metrics
        for metric in self.logged_metrics:
            results_str += (
                f'{metric.agg_metric_field}: {log[metric.agg_metric_field]:.3f}\n'
            )

        # Process logs for each group
        if not self.no_group_logging:
            for g in range(self.grouper.n_groups):
                group_count = log[f"count_group:{g}"]
                if group_count <= 0:
                    continue

                results_str += (
                    f'  {self.grouper.group_str(g)}  '
                    f'[n = {group_count:6.0f}]:\t'
                )

                # Process grouped logged fields
                for field in self.logged_fields:
                    if field.startswith(self.group_prefix):
                        field_suffix = field[len(self.group_prefix):]
                        log_key = f'{field_suffix}_group:{g}'
                        results_str += (
                            f'{field_suffix}: '
                            f'{log[log_key]:5.3f}\t'
                        )

                # Process grouped metric fields
                for metric in self.logged_metrics:
                    results_str += (
                        f'{metric.name}: '
                        f'{log[metric.group_metric_field(g)]:5.3f}\t'
                    )
                results_str += '\n'
        else:
            results_str += '\n'

        return results_str
