import torch
import torch.nn as nn

from configs.supported import process_pseudolabels_functions
from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from utils import move_to, collate_list, concat_input


class DropoutModel(nn.Module):
    def __init__(self, featurizer, classifier, dropout_rate):
        super().__init__()
        self.featurizer = featurizer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = classifier
        self.needs_y = featurizer.needs_y

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

    def __init__(
        self, config, d_out, grouper, loss, unlabeled_loss, metric, n_train_steps
    ):
        # initialize student model with dropout before last layer
        if config.noisystudent_add_dropout:
            featurizer, classifier = initialize_model(
                config, d_out=d_out, is_featurizer=True
            )
            student_model = DropoutModel(
                featurizer, classifier, config.noisystudent_dropout_rate
            ).to(config.device)
        else:
            student_model = initialize_model(config, d_out=d_out, is_featurizer=False)
        self.process_pseudolabels_function = process_pseudolabels_functions[
            config.process_pseudolabels_function
        ]

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
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (x, y, m): a batch of data yielded by data loaders
            - unlabeled_batch: examples (x, y_pseudo, m) where y_pseudo is an already-computed teacher pseudolabel
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pseudo (Tensor): pseudolabels for unlabeled batch (from loader)
                - unlabeled_y_pred (Tensor): model output on unlabeled batch
        """
        # Labeled examples
        x, y_true, metadata = labeled_batch
        n_lab = len(metadata)
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        # package the results
        results = {"g": g, "y_true": y_true, "metadata": metadata}

        # Unlabeled examples with pseudolabels
        if unlabeled_batch is not None:
            x_unlab, y_pseudo, metadata_unlab = unlabeled_batch
            x_unlab = move_to(x_unlab, self.device)
            g_unlab = move_to(self.grouper.metadata_to_group(metadata_unlab), self.device)
            y_pseudo = move_to(y_pseudo, self.device)
            results["unlabeled_metadata"] = metadata_unlab
            results["unlabeled_y_pseudo"] = y_pseudo
            results["unlabeled_g"] = g_unlab

            x_cat = concat_input(x, x_unlab)
            y_cat = collate_list([y_true, y_pseudo]) if self.model.needs_y else None
            outputs = self.get_model_output(x_cat, y_cat)
            results["y_pred"] = outputs[:n_lab]
            results["unlabeled_y_pred"] = outputs[n_lab:]
        else:
            results["y_pred"] = self.get_model_output(x, y_true)

        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )

        # Pseudolabel loss
        if "unlabeled_y_pseudo" in results:
            consistency_loss = self.unlabeled_loss.compute(
                results["unlabeled_y_pred"],
                results["unlabeled_y_pseudo"],
                return_dict=False,
            )
        else:
            consistency_loss = 0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(results, "consistency_loss", consistency_loss)

        return classification_loss + consistency_loss
