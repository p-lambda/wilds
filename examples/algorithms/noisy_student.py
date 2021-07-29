import torch
import torch.nn as nn
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from optimizer import initialize_optimizer_with_model_params
from wilds.common.utils import split_into_groups
from configs.supported import process_outputs_functions
import copy
from utils import load
import re

class DropoutModel(nn.Module):
    def __init__(self, featurizer, classifier, dropout_rate):
        super().__init__()
        self.featurizer = featurizer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = classifier
    def forward(self, x):
        features = self.featurizer(x)
        features_sparse = self.dropout(features)
        return self.classifier(features_sparse)

class NoisyStudent(SingleModelAlgorithm):
    """
    Noisy Student.
    This algorithm was originally proposed as a semi-supervised learning algorithm.

    One run of this codebase gives us one iteration (load a teacher, train student). To run another iteration,
    re-run the previous command, pointing config.teacher_model_path to the trained student weights.

    To warm start the student model, point config.pretrained_model_path to config.teacher_model_path

    Based on the original paper, loss is of the form
        \ell_s + \ell_u
    where 
        \ell_s = cross-entropy with true labels; student predicts with noise
        \ell_u = cross-entropy with pseudolabel generated without noise; student predicts with noise
    The student is noised using:
        - Input images are augmented using RandAugment
        - Single dropout layer before final classifier (fc) layer
        - TODO: stochastic depth with linearly decaying survival probability from last to first

    This code only supports hard pseudolabeling and a teacher that is the same class as the student (e.g. both densenet121s)

    Original paper:
        @inproceedings{xie2020self,
            title={Self-training with noisy student improves imagenet classification},
            author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
            pages={10687--10698},
            year={2020}
            }
    """
    def __init__(self, config, d_out, grouper, loss, unlabeled_loss, metric, n_train_steps):
        # initialize student model with dropout before last layer
        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        student_model = DropoutModel(featurizer, classifier, config.dropout_rate).to(config.device)

        # parameters_to_optimize: List[Dict] = [
        #     {"params": featurizer.parameters(), "lr": config.featurizer_lr},
        #     {"params": classifier.parameters(), "lr": config.classifier_lr},
        # ]
        # self.optimizer = initialize_optimizer_with_model_params(config, parameters_to_optimize)

        # initialize module
        super().__init__(
            config=config,
            model=student_model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.unlabeled_loss = unlabeled_loss
        # additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("consistency_loss")
        
    def process_batch(self, labeled_batch, unlabeled_batch=None):
        # Labeled examples
        x, y_true, metadata = labeled_batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        outputs = self.model(x)
        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata
        }
        # Unlabeled examples
        if unlabeled_batch is not None:
            x, y_pseudo, metadata = unlabeled_batch # x should be strongly augmented
            x = x.to(self.device)
            g = self.grouper.metadata_to_group(metadata).to(self.device)
            y_pseudo = y_pseudo.to(self.device)
            outputs = self.model(x)
            results['unlabeled_metadata'] = metadata
            results['unlabeled_y_pseudo'] = y_pseudo 
            results['unlabeled_y_pred'] = outputs
            results['unlabeled_g'] = g
        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

        # Pseudolabel loss
        if 'unlabeled_y_pred' in results: 
            consistency_loss = self.unlabeled_loss.compute(
                results['unlabeled_y_pred'], 
                results['unlabeled_y_pseudo'], 
                return_dict=False
            )
        else: 
            consistency_loss = 0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "consistency_loss", consistency_loss
        )

        return classification_loss + consistency_loss 
