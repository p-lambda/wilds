import torch

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

try:
    from torch_geometric.data import Batch
except ImportError:
    pass


class AFN(SingleModelAlgorithm):
    """
    Adaptive Feature Norm (AFN)

    @InProceedings{Xu_2019_ICCV,
        author = {Xu, Ruijia and Li, Guanbin and Yang, Jihan and Lin, Liang},
        title = {Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation},
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
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
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

        # Additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("afn_loss")

    def afn_loss(self, features):
        """
        Adapted from https://github.com/jihanyang/AFN
        """
        radius = features.norm(p=2, dim=1).detach()
        assert not radius.requires_grad
        radius = radius + 1.0
        loss = ((features.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Override
        """
        # Forward pass
        x, y_true, metadata = batch
        g = self.grouper.metadata_to_group(metadata).to(self.device)

        if unlabeled_batch is not None:
            unlabeled_x, unlabeled_metadata = unlabeled_batch

            # Concatenate examples and true domains
            if isinstance(x, torch.Tensor):
                x_cat = torch.cat((x, unlabeled_x), dim=0)
            elif isinstance(x, Batch):
                x.y = None
                x_cat = Batch.from_data_list([x, unlabeled_x])
            else:
                raise TypeError("x must be Tensor or Batch")
        else:
            x_cat = x

        x_cat = x_cat.to(self.device)
        y_true = y_true.to(self.device)

        features = self.featurizer(x_cat)
        y_pred = self.classifier(features)
        # Ignore the predicted labels for the unlabeled data
        y_pred = y_pred[: len(y_true)]

        return {
            "g": g,
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "features": features,
        }

    def objective(self, results):
        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )

        if self.is_training:
            features = results.pop("features")
            features_source = features[: len(results["y_true"])]
            features_target = features[len(results["y_true"]) :]
            afn_loss = self.afn_loss(features_source) + self.afn_loss(features_target)
        else:
            afn_loss = 0.0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(results, "afn_loss", afn_loss)
        return classification_loss + self.penalty_weight * afn_loss
