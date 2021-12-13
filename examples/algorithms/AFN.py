import torch

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

class AFN(SingleModelAlgorithm):
    """
    Adaptive Feature Norm (AFN)

    Original paper:
        @InProceedings{Xu_2019_ICCV,
            author = {Xu, Ruijia and Li, Guanbin and Yang, Jihan and Lin, Liang},
            title = {Larger Norm More Transferable: An Adaptive Feature Norm Approach for
                     Unsupervised Domain Adaptation},
            booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
            month = {October},
            year = {2019}
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
    ):
        # Initialize model
        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        model = torch.nn.Sequential(featurizer, classifier)

        # Initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

        # Model components
        self.featurizer = featurizer
        self.classifier = classifier

        # Algorithm hyperparameters
        self.penalty_weight = config.afn_penalty_weight
        self.delta_r = config.safn_delta_r
        self.r = config.hafn_r
        self.afn_loss = self.hafn_loss if config.use_hafn else self.safn_loss

        # Additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("feature_norm_penalty")

    def safn_loss(self, features):
        """
        Adapted from https://github.com/jihanyang/AFN
        """
        radius = features.norm(p=2, dim=1).detach()
        assert not radius.requires_grad
        radius = radius + self.delta_r
        loss = ((features.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss

    def hafn_loss(self, features):
        """
        Adapted from https://github.com/jihanyang/AFN
        """
        loss = (features.norm(p=2, dim=1).mean() - self.r) ** 2
        return loss

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - features (Tensor): featurizer output for batch
                - y_pred (Tensor): full model output for batch 
                - unlabeled_features (Tensor): featurizer outputs for unlabeled_batch
        """
        # Forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        features = self.featurizer(x)
        y_pred = self.classifier(features)

        results = {
            "g": g,
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "features": features,
        }

        if unlabeled_batch is not None:
            unlabeled_x, _ = unlabeled_batch
            unlabeled_x = unlabeled_x.to(self.device)
            results['unlabeled_features'] = self.featurizer(unlabeled_x)
        return results

    def objective(self, results):
        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )

        if self.is_training:
            f_source = results.pop("features")
            f_target = results.pop("unlabeled_features")
            feature_norm_penalty = self.afn_loss(f_source) + self.afn_loss(f_target)
        else:
            feature_norm_penalty = 0.0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "feature_norm_penalty", feature_norm_penalty
        )
        return classification_loss + self.penalty_weight * feature_norm_penalty