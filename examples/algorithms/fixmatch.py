from typing import Dict, List

import torch
import torch.nn.functional as F

from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from configs.supported import process_outputs_functions
from optimizer import initialize_optimizer_with_model_params

class FixMatch(SingleModelAlgorithm):
    """
    FixMatch.
    This algorithm was originally proposed as a semi-supervised learning algorithm. 

    Loss is of the form
        \ell_s + \lambda * \ell_u
    where 
        \ell_s = cross-entropy with true labels using weakly augmented labeled examples
        \ell_u = cross-entropy with pseudolabel generated using weak augmentation and prediction
            using strong augmentation

    Original paper:
        @article{sohn2020fixmatch,
            title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
            author={Sohn, Kihyuk and Berthelot, David and Li, Chun-Liang and Zhang, Zizhao and Carlini, Nicholas and Cubuk, Ekin D and Kurakin, Alex and Zhang, Han and Raffel, Colin},
            journal={arXiv preprint arXiv:2001.07685},
            year={2020}
            }
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        featurizer = featurizer.to(config.device)
        classifier = classifier.to(config.device)
        model = torch.nn.Sequential(featurizer, classifier).to(config.device)

        # TODO: change this to featurizer and classifier lr -Tony
        parameters_to_optimize: List[Dict] = [
            {"params": featurizer.parameters(), "lr": config.dann_featurizer_lr},
            {"params": classifier.parameters(), "lr": config.dann_classifier_lr},
        ]
        self.optimizer = initialize_optimizer_with_model_params(config, parameters_to_optimize)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.fixmatch_lambda = config.fixmatch_lambda
        self.confidence_threshold = config.fixmatch_confidence_threshold
        if config.process_outputs_function is not None: 
            self.process_outputs_function = process_outputs_functions[config.process_outputs_function]

        # Additional logging
        self.logged_fields.append("pseudolabels_kept_frac")
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("consistency_loss")

    def process_batch(self, labeled_batch, unlabeled_batch=None):
        """
        Args:
            - labeled_batch: examples (x, y, m) where x is weakly augmented
            - unlabeled_batch: examples (x, m) where x is unaugmented
        Returns: results, a dict containing keys:
            - 'g': groups for the labeled batch
            - 'y_true': true labels for the labeled batch
            - 'y_pred': outputs (logits) for the labeled batch
            - 'metadata': metdata tensor for the labeled batch
            - 'unlabeled_g': groups for the unlabeled batch
            - 'unlabeled_weak_y_pseudo': class pseudolabels predicted from weakly augmented x of the unlabeled batch
            - 'unlabeled_mask': true if the unlabeled example had confidence above the threshold; we pass this around 
                to help compute the loss in self.objective()
            - 'unlabeled_strong_y_pred': outputs (logits) on strongly augmented x of the unlabeled batch
            - 'unlabeled_metadata': metdata tensor for the unlabeled batch
        """
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
            x, metadata = unlabeled_batch
            x_weak, x_strong = x
            x_weak = x_weak.to(self.device)
            x_strong = x_strong.to(self.device)

            g = self.grouper.metadata_to_group(metadata).to(self.device)
            results['unlabeled_metadata'] = metadata
            results['unlabeled_g'] = g

            with torch.no_grad():
                outputs = self.model(x_weak)
                mask = torch.max(F.softmax(outputs, -1), -1)[0] >= self.confidence_threshold
                pseudolabels = self.process_outputs_function(outputs)
                results['unlabeled_weak_y_pseudo'] = pseudolabels
                results['unlabeled_mask'] = mask

            outputs = self.model(x_strong)
            results['unlabeled_strong_y_pred'] = outputs
        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

        # Pseudolabeled loss
        if 'unlabeled_weak_y_pseudo' in results:
            mask = results['unlabeled_mask']
            consistency_loss = self.loss.compute(
                results['unlabeled_strong_y_pred'][mask], 
                results['unlabeled_weak_y_pseudo'][mask], 
                return_dict=False
            )
            pseudolabels_kept_frac = mask.count_nonzero().item() / mask.shape[0]
        else:
            consistency_loss = 0
            pseudolabels_kept_frac = 0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "consistency_loss", consistency_loss
        )
        self.save_metric_for_logging(
            results, "pseudolabels_kept_frac", pseudolabels_kept_frac
        )

        return classification_loss + self.fixmatch_lambda * consistency_loss