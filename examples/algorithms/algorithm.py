import torch.nn as nn
from utils import move_to, detach_and_clone


class Algorithm(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.out_device = 'cpu'
        self._has_log = False
        self.reset_log()

    def update(self, batch):
        """
        Process the batch, update the log, and update the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
        """
        raise NotImplementedError

    def evaluate(self, batch):
        """
        Process the batch and update the log, without updating the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
        """
        raise NotImplementedError

    def train(self, mode=True):
        """
        Switch to train mode
        """
        self.is_training = mode
        super().train(mode)
        self.reset_log()

    @property
    def has_log(self):
        return self._has_log

    def reset_log(self):
        """
        Resets log by clearing out the internal log, Algorithm.log_dict
        """
        self._has_log = False
        self.log_dict = {}

    def update_log(self, results):
        """
        Updates the internal log, Algorithm.log_dict
        Args:
            - results (dictionary)
        """
        raise NotImplementedError

    def get_log(self):
        """
        Sanitizes the internal log (Algorithm.log_dict) and outputs it.

        """
        raise NotImplementedError

    def get_pretty_log_str(self):
        raise NotImplementedError

    def step_schedulers(self, is_epoch, metrics={}, log_access=False):
        """
        Update all relevant schedulers
        Args:
            - is_epoch (bool): epoch-wise update if set to True, batch-wise update otherwise
            - metrics (dict): a dictionary of metrics that can be used for scheduler updates
            - log_access (bool): whether metrics from self.get_log() can be used to update schedulers
        """
        raise NotImplementedError

    def sanitize_dict(self, in_dict, to_out_device=True):
        """
        Helper function that sanitizes dictionaries by:
            - moving to the specified output device
            - removing any gradient information
            - detaching and cloning the tensors
        Args:
            - in_dict (dictionary)
        Output:
            - out_dict (dictionary): sanitized version of in_dict
        """
        out_dict = detach_and_clone(in_dict)
        if to_out_device:
            out_dict = move_to(out_dict, self.out_device)
        return out_dict