import numpy as np
import torch

from models.initializer import initialize_model, initialize_domain_adversarial_network
from algorithms.single_model_algorithm import SingleModelAlgorithm


class DANN(SingleModelAlgorithm):
    """
    Domain-adversarial neural network.

    Original paper:
        @inproceedings{dann,
          title={Domain-Adversarial Training of Neural Networks},
          author={Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand and Lempitsky},
          booktitle={Journal of Machine Learning Research 17},
          year={2016}
        }
    """

    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        # Initialize model - standard feed forward architecture
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        featurizer = featurizer.to(config.device)
        classifier = classifier.to(config.device)
        model = initialize_domain_adversarial_network(featurizer, classifier).to(
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
        # The authors used binomial cross-entropy loss to calculate the domain loss
        self.domain_loss = torch.nn.functional.binary_cross_entropy

        # The authors set this value to 10. for their experiments in the paper
        self.gamma = config.dann_gamma

        # Additional logging
        self.logged_fields.append("grl_lambda")
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("domain_classification_loss")

    def process_batch(self, batch, unlabeled_batch=None, batch_info=None):
        """
        Override
        """
        if batch_info is not None:
            # p is the training progress linearly changing from 0 to 1, so to normalize
            # make the numerator be the total number of batches trained so far
            # and the denominator be the total number of batches.
            p = float(
                batch_info["epoch"] * batch_info["n_batches"] + batch_info["batch"]
            ) / (batch_info["n_epochs"] * batch_info["n_batches"])
            # Authors set gamma to 10 for all their experiments
            # Calculate lambda for the gradient reverse layer
            grl_lambda = (2 / (1 + np.exp(-self.gamma * p))) - 1
        else:
            grl_lambda = 0.0

        # Forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        domains_true = torch.zeros(len(x), 1).to(self.device)
        y_pred, domains_pred = self.model(x, grl_lambda)

        results = {
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "domains_true": domains_true,
            "domains_pred": domains_pred,
            "grl_lambda": grl_lambda,
        }

        if unlabeled_batch is not None:
            unlabeled_x, _ = unlabeled_batch
            unlabeled_x = unlabeled_x.to(self.device)
            unlabeled_domains_true = torch.ones(len(unlabeled_x), 1).to(self.device)
            _, unlabeled_domains_pred = self.model(unlabeled_x, grl_lambda)
            results["unlabeled_domains_true"] = unlabeled_domains_true
            results["unlabeled_domains_pred"] = unlabeled_domains_pred

        return results

    def objective(self, results):
        def log_metric(metric, value):
            if isinstance(value, torch.Tensor):
                results[metric] = value.item()
            else:
                results[metric] = value

        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )

        if self.is_training:
            domain_classification_loss = self.domain_loss(
                results.pop("domains_pred"),
                results.pop("domains_true"),
            )
            domain_classification_loss += self.domain_loss(
                results.pop("unlabeled_domains_pred"),
                results.pop("unlabeled_domains_true"),
            )
        else:
            domain_classification_loss = 0.0

        # Add to results for additional logging
        log_metric("classification_loss", classification_loss)
        log_metric("domain_classification_loss", domain_classification_loss)

        return classification_loss + domain_classification_loss
