import numpy as np
import torch

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.domain_adversarial_network import DomainAdversarialNetwork
from models.initializer import initialize_model


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
        model = DomainAdversarialNetwork(featurizer, classifier, n_domains).to(
            config.device
        )

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
        self.gamma = config.dann_gamma
        self.penalty_weight = config.dann_penalty_weight

        # Additional logging
        self.logged_fields.append("grl_lambda")
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("domain_classification_loss")

    def process_batch(self, batch, unlabeled_batch=None, batch_info=None):
        """
        Override
        """
        if batch_info is not None:
            # TODO: test the impact of the domain adaptation parameter lambda.
            #       If it improves performance, use scheduler instead.
            # p is the training progress linearly changing from 0 to 1, so to normalize
            # make the numerator be the total number of batches trained so far
            # and the denominator be the total number of batches.
            p = float(
                batch_info["epoch"] * batch_info["n_batches"] + batch_info["batch"]
            ) / (batch_info["n_epochs"] * batch_info["n_batches"])
            # Calculate lambda for the gradient reverse layer
            grl_lambda = (2 / (1 + np.exp(-self.gamma * p))) - 1
        else:
            grl_lambda = 0.0

        # Forward pass
        x, y_true, metadata = batch
        domains_true = self.group_ids_to_domains[
            self.grouper.metadata_to_group(metadata)
        ]

        if unlabeled_batch is not None:
            unlabeled_x, unlabeled_metadata = unlabeled_batch
            unlabeled_domains_true = self.group_ids_to_domains[
                self.grouper.metadata_to_group(unlabeled_metadata)
            ]

            # Concatenate examples and true domains
            x = torch.cat([x, unlabeled_x])
            domains_true = torch.cat([domains_true, unlabeled_domains_true])

        x = x.to(self.device)
        y_true = y_true.to(self.device)
        domains_true = domains_true.to(self.device)
        y_pred, domains_pred = self.model(x, grl_lambda)

        # Ignore the predicted labels for the unlabeled data
        y_pred = y_pred[: len(y_true)]

        return {
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "domains_true": domains_true,
            "domains_pred": domains_pred,
            "grl_lambda": grl_lambda,
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
