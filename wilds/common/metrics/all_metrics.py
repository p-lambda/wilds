import copy

import numpy as np
import torch

import torch.nn.functional as F
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
from torchvision.ops import nms, box_convert
from wilds.common.metrics.metric import Metric, ElementwiseMetric, MultiTaskMetric
from wilds.common.metrics.loss import ElementwiseLoss
from wilds.common.utils import avg_over_groups, minimum, maximum, get_counts
import sklearn.metrics
from scipy.stats import pearsonr

def binary_logits_to_score(logits):
    assert logits.dim() in (1,2)
    if logits.dim()==2: #multi-class logits
        assert logits.size(1)==2, "Only binary classification"
        score = F.softmax(logits, dim=1)[:,1]
    else:
        score = logits
    return score

def multiclass_logits_to_pred(logits):
    """
    Takes multi-class logits of size (batch_size, ..., n_classes) and returns predictions
    by taking an argmax at the last dimension
    """
    assert logits.dim() > 1
    return logits.argmax(-1)

def binary_logits_to_pred(logits):
    return (logits>0).long()

def pseudolabel_binary_logits(logits, confidence_threshold):
    """
    Input:
        logits (Tensor): Binary logits of size (batch_size, n_tasks).
                         If an entry is >0, it means the prediction for taht
                         (example, task) is positive.
        confidence_threshold (float): In [0,1]

    Output:
        unlabeled_y_pred (Tensor): Filtered version of logits, discarding any rows (examples) that
                                   have no predictions with confidence above confidence_threshold.
        unlabeled_y_pseudo (Tensor): Corresponding hard-pseudo-labeled version of logits. All
                                     entries with confidence below confidence_threshold are set to
                                     nan. All rows with no confident entries are discarded.
        pseudolabels_kept_frac (float): Fraction of (examples, tasks) not set to nan or discarded.
        mask (Tensor): Mask used to discard predictions with confidence under the confidence threshold.
    """
    if len(logits.shape) != 2:
        raise ValueError('Logits must be 2-dimensional.')
    probs = 1 / (1 + torch.exp(-logits))
    mask = (torch.max(probs, 1-probs) >= confidence_threshold)
    unlabeled_y_pseudo = (logits > 0).float()
    unlabeled_y_pseudo[~mask] = float('nan')
    pseudolabels_kept_frac = mask.sum() / mask.numel() # mask is bool, so no .mean()
    example_mask = torch.any(~torch.isnan(unlabeled_y_pseudo), dim=1)
    unlabeled_y_pseudo = unlabeled_y_pseudo[example_mask]
    unlabeled_y_pred = logits[example_mask]
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, example_mask

def pseudolabel_multiclass_logits(logits, confidence_threshold):
    """
    Input:
        logits (Tensor): Multi-class logits of size (batch_size, ..., n_classes).
        confidence_threshold (float): In [0,1]

    Output:
        unlabeled_y_pred (Tensor): Filtered version of logits, discarding any rows (examples) that
                                   have no predictions with confidence above confidence_threshold.
        unlabeled_y_pseudo (Tensor): Corresponding hard-pseudo-labeled version of logits. All
                                     examples with confidence below confidence_threshold are discarded.
        pseudolabels_kept_frac (float): Fraction of examples not discarded.
        mask (Tensor): Mask used to discard predictions with confidence under the confidence threshold.
    """
    mask = torch.max(F.softmax(logits, -1), -1)[0] >= confidence_threshold
    unlabeled_y_pseudo = multiclass_logits_to_pred(logits)
    unlabeled_y_pseudo = unlabeled_y_pseudo[mask]
    unlabeled_y_pred = logits[mask]
    pseudolabels_kept_frac = mask.sum() / mask.numel() # mask is bool, so no .mean()
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, mask

def pseudolabel_identity(logits, confidence_threshold):
    return logits, logits, 1, None

def pseudolabel_detection(preds, confidence_threshold):
    """
    Input:
        preds (List): List of len batch_size. Each entry is a dict containing
                      the keys 'boxes', 'labels', 'scores', and 'losses'
                      ('losses' is empty)
        confidence_threshold (float): In [0,1]
    """
    preds, pseudolabels_kept_frac = _mask_pseudolabels_detection(preds, confidence_threshold)
    unlabeled_y_pred = [
        {
            'boxes': pred['boxes'],
            'labels': pred['labels'],
            'scores': pred['scores'],
            'losses': pred['losses'],
        } for pred in preds
    ]
    unlabeled_y_pseudo = [
        {
            'boxes': pred['boxes'],
            'labels': pred['labels'],
        } for pred in preds
    ]

    # Keep all examples even if they don't have any confident-enough predictions
    # They will be treated as empty images
    example_mask = torch.ones(len(preds), dtype=torch.bool)
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, example_mask


def pseudolabel_detection_discard_empty(preds, confidence_threshold):
    """
    Input:
        preds (List): List of len batch_size. Each entry is a dict containing
                      the keys 'boxes', 'labels', 'scores', and 'losses'
                      ('losses' is empty)
        confidence_threshold (float): In [0,1]
    """
    preds, pseudolabels_kept_frac = _mask_pseudolabels_detection(preds, confidence_threshold)
    unlabeled_y_pred = [
        {
            'boxes': pred['boxes'],
            'labels': pred['labels'],
            'scores': pred['scores'],
            'losses': pred['losses'],
        } for pred in preds if len(pred['labels']) > 0
    ]
    unlabeled_y_pseudo = [
        {
            'boxes': pred['boxes'],
            'labels': pred['labels'],
        } for pred in preds if len(pred['labels']) > 0
    ]
    example_mask = torch.tensor([len(pred['labels']) > 0 for pred in preds])
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, example_mask


def _mask_pseudolabels_detection(preds, confidence_threshold):
    total_boxes = 0.0
    kept_boxes = 0.0

    preds = copy.deepcopy(preds)
    for pred in preds:
        mask = (pred['scores'] >= confidence_threshold)
        pred['boxes'] = pred['boxes'][mask]
        pred['labels'] = pred['labels'][mask]
        pred['scores'] = pred['scores'][mask]
        total_boxes += len(mask)
        kept_boxes += mask.sum()

    pseudolabels_kept_frac = kept_boxes / total_boxes
    return preds, pseudolabels_kept_frac


class Accuracy(ElementwiseMetric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        return (y_pred==y_true).float()

    def worst(self, metrics):
        return minimum(metrics)

class MultiTaskAccuracy(MultiTaskMetric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn  # should work on flattened inputs
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if self.prediction_fn is not None:
            flattened_y_pred = self.prediction_fn(flattened_y_pred)
        return (flattened_y_pred==flattened_y_true).float()

    def worst(self, metrics):
        return minimum(metrics)

class MultiTaskAveragePrecision(MultiTaskMetric):
    def __init__(self, prediction_fn=None, name=None, average='macro'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'avgprec'
            if average is not None:
                name+=f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if self.prediction_fn is not None:
            flattened_y_pred = self.prediction_fn(flattened_y_pred)
        ytr = np.array(flattened_y_true.squeeze().detach().cpu().numpy() > 0)
        ypr = flattened_y_pred.squeeze().detach().cpu().numpy()
        score = sklearn.metrics.average_precision_score(
            ytr,
            ypr,
            average=self.average
        )
        to_ret = torch.tensor(score).to(flattened_y_pred.device)
        return to_ret

    def _compute_group_wise(self, y_pred, y_true, g, n_groups):
        group_metrics = []
        group_counts = get_counts(g, n_groups)
        for group_idx in range(n_groups):
            if group_counts[group_idx]==0:
                group_metrics.append(torch.tensor(0., device=g.device))
            else:
                flattened_metrics, _ = self.compute_flattened(
                    y_pred[g == group_idx],
                    y_true[g == group_idx],
                    return_dict=False)
                group_metrics.append(flattened_metrics)
        group_metrics = torch.stack(group_metrics)
        worst_group_metric = self.worst(group_metrics[group_counts>0])

        return group_metrics, group_counts, worst_group_metric

    def worst(self, metrics):
        return minimum(metrics)

class Recall(Metric):
    def __init__(self, prediction_fn=None, name=None, average='binary'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'recall'
            if average is not None:
                name+=f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred, average=self.average, labels=torch.unique(y_true))
        return torch.tensor(recall)

    def worst(self, metrics):
        return minimum(metrics)

class F1(Metric):
    def __init__(self, prediction_fn=None, name=None, average='binary'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'F1'
            if average is not None:
                name+=f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        score = sklearn.metrics.f1_score(y_true, y_pred, average=self.average, labels=torch.unique(y_true))
        return torch.tensor(score)

    def worst(self, metrics):
        return minimum(metrics)

class PearsonCorrelation(Metric):
    def __init__(self, name=None):
        if name is None:
            name = 'r'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        r = pearsonr(y_pred.squeeze().detach().cpu().numpy(), y_true.squeeze().detach().cpu().numpy())[0]
        return torch.tensor(r)

    def worst(self, metrics):
        return minimum(metrics)

def mse_loss(out, targets):
    assert out.size()==targets.size()
    if out.numel()==0:
        return torch.Tensor()
    else:
        assert out.dim()>1, 'MSE loss currently supports Tensors of dimensions > 1'
        losses = (out - targets)**2
        reduce_dims = tuple(list(range(1, len(targets.shape))))
        losses = torch.mean(losses, dim=reduce_dims)
        return losses

class MSE(ElementwiseLoss):
    def __init__(self, name=None):
        if name is None:
            name = 'mse'
        super().__init__(name=name, loss_fn=mse_loss)

class PrecisionAtRecall(Metric):
    """Given a specific model threshold, determine the precision score achieved"""
    def __init__(self, threshold, score_fn=None, name=None):
        self.score_fn = score_fn
        self.threshold = threshold
        if name is None:
            name = "precision_at_global_recall"
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        score = self.score_fn(y_pred)
        predictions = (score > self.threshold)
        return torch.tensor(sklearn.metrics.precision_score(y_true, predictions))

    def worst(self, metrics):
        return minimum(metrics)

class DummyMetric(Metric):
    """
    For testing purposes. This Metric always returns -1.
    """
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'dummy'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        return torch.tensor(-1)

    def _compute_group_wise(self, y_pred, y_true, g, n_groups):
        group_metrics = torch.ones(n_groups, device=g.device) * -1
        group_counts = get_counts(g, n_groups)
        worst_group_metric = self.worst(group_metrics)
        return group_metrics, group_counts, worst_group_metric

    def worst(self, metrics):
        return minimum(metrics)

class DetectionAccuracy(ElementwiseMetric):
    """
    Given a specific Intersection over union threshold,
    determine the accuracy achieved for a one-class detector
    """

    def __init__(self, iou_threshold=0.5, score_threshold=0.5, name=None):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        if name is None:
            name = "detection_acc"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        batch_results = []
        for src_boxes, target in zip(y_true, y_pred):
            target_boxes = target["boxes"]
            target_scores = target["scores"]

            pred_boxes = target_boxes[target_scores > self.score_threshold]
            det_accuracy = torch.mean(torch.stack([ self._accuracy(src_boxes["boxes"],pred_boxes,iou_thr) for iou_thr in np.arange(0.5,0.51,0.05)]))
            batch_results.append(det_accuracy)

        return torch.tensor(batch_results)

    def _accuracy(self, src_boxes,pred_boxes ,  iou_threshold):
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on iou
            matcher = Matcher(
                iou_threshold,
                iou_threshold,
                allow_low_quality_matches=False)
            match_quality_matrix = box_iou(
                src_boxes,
                pred_boxes)
            results = matcher(match_quality_matrix)
            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]
            # in Matcher, a pred element can be matched only twice
            false_positive = (
                torch.count_nonzero(results == -1) +
                (len(matched_elements) - len(matched_elements.unique()))
            )
            false_negative = total_gt - true_positive
            acc = true_positive / ( true_positive + false_positive + false_negative )
            return true_positive / ( true_positive + false_positive + false_negative )
        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.)
            else:
                return torch.tensor(1.)
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.)

    def worst(self, metrics):
        return minimum(metrics)
