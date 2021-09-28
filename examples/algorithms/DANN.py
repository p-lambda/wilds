from typing import Dict, List

import torch

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.domain_adversarial_network import DomainAdversarialNetwork
from models.initializer import initialize_model
from optimizer import initialize_optimizer_with_model_params

try:
    from torch_geometric.data import Batch
except ImportError:
    pass

class DANN(SingleModelAlgorithm):
    """
    Domain-adversarial training of neural networks.

    Original paper:
        @inproceedings{dann,
          title={Domain-Adversarial Training of Neural Networks},
          author={Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand and Lempitsky},
          booktitle={Journal of Machine Learning Research 17},
          year={2016}
        }
    """

    def __init__(
        self,
        config,
        d_out,
        grouper,
        loss,
        metric,
        n_train_steps,
        n_domains,
        group_ids_to_domains,
    ):
        # Initialize model
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        model = DomainAdversarialNetwork(featurizer, classifier, n_domains)
        parameters_to_optimize: List[Dict] = model.get_parameters_with_lr(
            featurizer_lr=config.dann_featurizer_lr,
            classifier_lr=config.dann_classifier_lr,
            discriminator_lr=config.dann_discriminator_lr,
        )
        self.optimizer = initialize_optimizer_with_model_params(config, parameters_to_optimize)

        # Initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.group_ids_to_domains = group_ids_to_domains

        # Algorithm hyperparameters
        self.penalty_weight = config.dann_penalty_weight

        # Additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("domain_classification_loss")

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Override
        """
        # Forward pass
        x, y_true, metadata = batch
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        domains_true = self.group_ids_to_domains[g]

        if unlabeled_batch is not None:
            unlabeled_x, unlabeled_metadata = unlabeled_batch
            unlabeled_domains_true = self.group_ids_to_domains[
                self.grouper.metadata_to_group(unlabeled_metadata)
            ]

            # Concatenate examples and true domains
            if isinstance(x, torch.Tensor):
                x_cat = torch.cat((x, unlabeled_x), dim=0)
            elif isinstance(x, Batch):
                x.y = None
                x_cat = Batch.from_data_list([x, unlabeled_x])
            else:
                raise TypeError('x must be Tensor or Batch')

            domains_true = torch.cat([domains_true, unlabeled_domains_true])

        x_cat = x_cat.to(self.device)
        y_true = y_true.to(self.device)
        domains_true = domains_true.to(self.device)
        y_pred, domains_pred = self.model(x_cat)

        # Ignore the predicted labels for the unlabeled data
        y_pred = y_pred[: len(y_true)]

        return {
            "g": g,
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "domains_true": domains_true,
            "domains_pred": domains_pred,
        }

    def objective(self, results):
        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )

        if self.is_training:
            domain_classification_loss = self.loss.compute(
                results.pop("domains_pred"),
                results.pop("domains_true"),
                return_dict=False,
            )
        else:
            domain_classification_loss = 0.0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "domain_classification_loss", domain_classification_loss
        )

        return classification_loss + domain_classification_loss * self.penalty_weight
