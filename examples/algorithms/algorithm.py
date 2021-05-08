import torch
import torch.nn as nn

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

    # Taken from domainbed
    def train(self, mode=True):
        """
        Switch to train mode
        """
        self.is_training = mode
        super().train(mode)
        torch.set_grad_enabled(mode)
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
            - turning any Tensor of size 1 to a simple number
        Args:
            - in_dict (dictionary)
        Output:
            - out_dict (dictionary): sanitized version of in_dict
        """
        out_dict = {}
        for k, v in in_dict.items():
            if isinstance(v, torch.Tensor):
                v_out = v.detach().clone()
                if to_out_device:
                    v_out = v_out.to(self.out_device)
            else:
                v_out = v
            out_dict[k] = v_out
        return out_dict
