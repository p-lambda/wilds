import torch.nn as nn
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import MSE

def initialize_loss(config, d_out):
    if config.loss_function == 'cross_entropy':
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none'))

    elif config.loss_function == 'lm_cross_entropy':
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none'))

    elif config.loss_function == 'mse':
        return MSE(name='loss')

    elif config.loss_function == 'multitask_bce':
        return MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

    elif config.loss_function == 'fasterrcnn_criterion':
        from models.detection.fasterrcnn import FasterRCNNLoss
        return ElementwiseLoss(loss_fn=FasterRCNNLoss(config.device))

    else:
        raise ValueError(f'config.loss_function {config.loss_function} not recognized')
