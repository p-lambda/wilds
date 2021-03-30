import torch.nn as nn
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss

def initialize_loss(config, d_out):
    if config.loss_function == 'cross_entropy':
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none'))

    elif config.loss_function == 'lm_cross_entropy':
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none'))

    elif config.loss_function == 'MSE':
        return MSE(name='loss')

    elif config.loss_function == 'multitask_bce':
        return MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

    elif config.loss_function == 'detr_set_criterion':
        return ElementwiseLoss(loss_fn=get_detr_set_criterion(config, d_out))        
    elif config.loss_function == 'faster_criterion':
        return ElementwiseLoss(loss_fn=get_faster_criterion(config))        

    else:
        raise ValueError(f'config.loss_function {config.loss_function} not recognized')


def get_faster_criterion(config):
    from examples.models.detection.fasterrcnn import FasterRCNNLoss

    criterion = FasterRCNNLoss(config.device)
    return criterion


def get_detr_set_criterion(config, d_out):
    from examples.models.detr.matcher import HungarianMatcher
    from examples.models.detr.detr import SetCriterion

    matcher = HungarianMatcher(
        cost_class=config.loss_kwargs['set_cost_class'],
        cost_bbox=config.loss_kwargs['set_cost_bbox'],
        cost_giou=config.loss_kwargs['set_cost_giou'])
    weight_dict = {
        'loss_ce': 1,
        'loss_bbox': config.loss_kwargs['bbox_loss_coef']}
    weight_dict['loss_giou'] = config.loss_kwargs['giou_loss_coef']

    if config.model_kwargs['aux_loss']:
        aux_weight_dict = {}
        for i in range(config.model_kwargs['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        d_out,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config.loss_kwargs['eos_coef'],
        losses=['labels', 'boxes', 'cardinality']).to(config.device)

    return criterion
