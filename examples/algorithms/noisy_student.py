import torch
import torch.nn as nn
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from optimizer import initialize_optimizer_with_model_params
from wilds.common.utils import split_into_groups

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
    We do not use stochastic depth.

    Pseudolabels are generated in run_expt.py on unlabeled images that have only been randomly cropped and flipped ("weak" transform).
    By default, we use hard pseudolabels; use the --soft_pseudolabels flag to add soft pseudolabels.

    This code only supports a teacher that is the same class as the student (e.g. both densenet121s)

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
        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata
        }

        # Unlabeled examples with pseudolabels
        if unlabeled_batch is not None:
            x_unlab, y_pseudo, metadata = unlabeled_batch # x should be strongly augmented
            x_unlab = x_unlab.to(self.device)
            g = self.grouper.metadata_to_group(metadata).to(self.device)
            y_pseudo = y_pseudo.to(self.device)
            results['unlabeled_metadata'] = metadata
            results['unlabeled_y_pseudo'] = y_pseudo 
            results['unlabeled_g'] = g

        # Concat and call forward
        n_lab = x.shape[0]
        if unlabeled_batch is not None: x_concat = torch.cat((x, x_unlab), dim=0)
        else: x_concat = x
        outputs = self.model(x_concat)
        results['y_pred'] = outputs[:n_lab]
        if unlabeled_batch is not None:
            results['unlabeled_y_pred'] = outputs[n_lab:]

        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

        # Pseudolabel loss
        if 'unlabeled_y_pseudo' in results: 
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
