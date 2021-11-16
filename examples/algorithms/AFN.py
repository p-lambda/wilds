import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from optimizer import initialize_optimizer_with_model_params

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
        featurizer, _ = initialize_model(config, d_out=d_out, is_featurizer=True)
        model = AFNModel(featurizer, d_out=d_out)
        parameters_to_optimize: List[Dict] = model.get_parameters()
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
        Override
        """
        # Forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        y_pred, features = self.model(x)

        results = {
            "g": g,
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "features": features,
        }

        if unlabeled_batch is not None:
            unlabeled_x, unlabeled_metadata = unlabeled_batch
            unlabeled_x = unlabeled_x.to(self.device)
            _, unlabeled_features = self.model(unlabeled_x)
            results['unlabeled_features'] = unlabeled_features
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


class AFNModel(nn.Module):
    def __init__(self, featurizer, d_out, bottleneck_dim: Optional[int] = 1000):
        super().__init__()
        self.featurizer = featurizer
        self.bottleneck = nn.Sequential(
            Block(featurizer.d_out, bottleneck_dim, dropout_p=0.5)
        )
        self.classifier = nn.Linear(bottleneck_dim, d_out)

        for m in self.bottleneck.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.01)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.normal_(0.0, 0.01)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.normal_(0.0, 0.01)

    def forward(self, x):
        features = self.featurizer(x)
        features = self.bottleneck(features)
        predictions = self.classifier(features)
        return predictions, features

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.featurizer.parameters()},
            {"params": self.bottleneck.parameters(), "momentum": 0.9},
            {"params": self.classifier.parameters(), "momentum": 0.9},
        ]
        return params

class Block(nn.Module):
    r"""
    Adapted from https://github.com/thuml/Transfer-Learning-Library.

    Basic building block for Image Classifier with structure: FC-BN-ReLU-Dropout.
    We use :math:`L_2` preserved dropout layers.
    Given mask probability :math:`p`, input :math:`x_k`, generated mask :math:`a_k`,
    vanilla dropout layers calculate

    .. math::
        \hat{x}_k = a_k\frac{1}{1-p}x_k\\

    While in :math:`L_2` preserved dropout layers

    .. math::
        \hat{x}_k = a_k\frac{1}{\sqrt{1-p}}x_k\\

    Args:
        in_features (int): Dimension of input features
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1000
        dropout_p (float, optional): dropout probability. Default: 0.5
    """

    def __init__(
        self,
        in_features: int,
        bottleneck_dim: Optional[int] = 1000,
        dropout_p: Optional[float] = 0.5,
    ):
        super(Block, self).__init__()
        self.fc = nn.Linear(in_features, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.fc(x)
        f = self.bn(f)
        f = self.relu(f)
        f = self.dropout(f)
        if self.training:
            f.mul_(math.sqrt(1 - self.dropout_p))
        return f