import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn

try:
    from torch_geometric.data import Batch
except ImportError:
    pass

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
        featurizer, _ = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        model = AFNModel(featurizer, d_out=d_out)

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

        # Additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("afn_loss")

    def afn_loss_stepwise(self, features):
        """
        Adapted from https://github.com/jihanyang/AFN
        """
        radius = features.norm(p=2, dim=1).detach()
        assert not radius.requires_grad
        radius = radius + 1.0
        loss = ((features.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss

    def afn_loss(self, features):
        """
        Adapted from https://github.com/jihanyang/AFN
        """
        loss = (features.norm(p=2, dim=1).mean() - 1) ** 2
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

        y_pred, features = self.model(x_cat)
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
            f_source = features[: len(results["y_true"])]
            f_target = features[len(results["y_true"]) :]
            afn_loss = self.afn_loss(f_source) + self.afn_loss(f_target)
        else:
            afn_loss = 0.0

        import pdb; pdb.set_trace()

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(results, "afn_loss", afn_loss)
        return classification_loss + self.penalty_weight * afn_loss


class AFNModel(nn.Module):
    def __init__(self, featurizer, d_out, bottleneck_dim: Optional[int] = 1000):
        super().__init__()
        self.featurizer = featurizer
        self.bottleneck = nn.Sequential(Block(featurizer.d_out, bottleneck_dim=bottleneck_dim))
        self.classifier = nn.Linear(bottleneck_dim, d_out)

    def forward(self, x):
        features = self.featurizer(x)
        features = self.bottleneck(features)
        predictions = self.classifier(features)
        return predictions, features


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
