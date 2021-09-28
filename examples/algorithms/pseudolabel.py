import torch
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from scheduler import LinearScheduleWithWarmupAndThreshold
from wilds.common.utils import split_into_groups
from configs.supported import process_outputs_functions, process_pseudolabels_functions
import copy
from utils import load

class PseudoLabel(SingleModelAlgorithm):
    """
    PseudoLabel.
    This is a vanilla pseudolabeling algorithm which updates the model per batch and incorporates a confidence threshold.

    Original paper:
        @inproceedings{lee2013pseudo,
            title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
            author={Lee, Dong-Hyun and others},
            booktitle={Workshop on challenges in representation learning, ICML},
            volume={3},
            number={2},
            pages={896},
            year={2013}
            }
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        model = initialize_model(config, d_out=d_out)
        model = model.to(config.device)
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
        self.lambda_scheduler = LinearScheduleWithWarmupAndThreshold(
            max_value=config.self_training_lambda,
            step_every_batch=True, # step per batch
            last_warmup_step=0,
            threshold_step=config.pseudolabel_T2 * n_train_steps
        )
        self.schedulers.append(self.lambda_scheduler)
        self.scheduler_metric_names.append(None)
        self.confidence_threshold = config.self_training_threshold
        if config.process_outputs_function is not None:
            self.process_outputs_function = process_outputs_functions[config.process_outputs_function]
        if config.process_pseudolabels_function is not None:
            self.process_pseudolabels_function = process_pseudolabels_functions[config.process_pseudolabels_function]
        # Additional logging
        self.logged_fields.append("pseudolabels_kept_frac")
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("consistency_loss")

    def process_batch(self, labeled_batch, unlabeled_batch=None):
        """
        Args:
            - labeled_batch: examples (x, y, m)
            - unlabeled_batch: examples (x, m)
        Returns: results, a dict containing keys:
            - 'g': groups for the labeled batch
            - 'y_true': true labels for the labeled batch
            - 'y_pred': outputs (logits) for the labeled batch
            - 'metadata': metdata tensor for the labeled batch
            - 'unlabeled_g': groups for the unlabeled batch
            - 'unlabeled_y_pseudo': class pseudolabels of the unlabeled batch
            - 'unlabeled_mask': true if the unlabeled example had confidence above the threshold; we pass this around
                to help compute the loss in self.objective()
            - 'unlabeled_y_pred': outputs (logits) on x of the unlabeled batch
            - 'unlabeled_metadata': metdata tensor for the unlabeled batch
        """
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
        # Unlabeled examples
        if unlabeled_batch is not None:
            x_unlab, metadata = unlabeled_batch
            x_unlab = x_unlab.to(self.device)
            g = self.grouper.metadata_to_group(metadata).to(self.device)
            results['unlabeled_metadata'] = metadata
            results['unlabeled_g'] = g

        # Concat and call forward
        if unlabeled_batch is not None:
            if isinstance(x, torch.Tensor):
                n_lab = x.shape[0]
                x_concat = torch.cat((x, x_unlab), dim=0)
                outputs = self.model(x_concat)
                results['y_pred'] = outputs[:n_lab]
                unlabeled_output = outputs[n_lab:]
            else:
                results['y_pred'] = self.model(x)
                unlabeled_output = self.model(x_unlab)

            unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac = self.process_pseudolabels_function(
                unlabeled_output,
                self.confidence_threshold)
            results['unlabeled_y_pred'] = unlabeled_y_pred
            results['unlabeled_y_pseudo'] = unlabeled_y_pseudo.detach().clone()

        else:
            results['y_pred'] = self.model(x)
            pseudolabels_kept_frac = 0

        self.save_metric_for_logging(
            results, "pseudolabels_kept_frac", pseudolabels_kept_frac
        )

        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss.compute(
            results['y_pred'],
            results['y_true'],
            return_dict=False)
        # Pseudolabeled loss
        if 'unlabeled_y_pseudo' in results:
            loss_output = self.loss.compute(
                results['unlabeled_y_pred'],
                results['unlabeled_y_pseudo'],
                return_dict=False,
            )
            consistency_loss = loss_output * results['pseudolabels_kept_frac']
        else:
            consistency_loss = 0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "consistency_loss", consistency_loss
        )
        # self.save_metric_for_logging(
        #     results, "pseudolabels_kept_frac", results['pseudolabels_kept_frac']
        # )

        return classification_loss + self.lambda_scheduler.value * consistency_loss
