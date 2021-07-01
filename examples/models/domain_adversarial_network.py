from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function


class DomainDiscriminator(nn.Sequential):
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_
    In the original paper and implementation, we distinguish whether the input features come
    from the source domain or the target domain.

    We extended this to work with multiple domains, which is controlled by the n_domains
    argument.

    Args:
        in_feature (int): dimension of the input feature
        n_domains (int): number of domains to discriminate
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, n_domains)`
    """

    def __init__(
        self, in_feature: int, n_domains, hidden_size: int = 1024, batch_norm=True
    ):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_domains),
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, n_domains),
            )

    def get_parameters(self, lr=1.0) -> List[Dict]:
        return [{"params": self.parameters(), "lr": lr}]

class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class DomainAdversarialNetwork(nn.Module):
    def __init__(self, featurizer, classifier, n_domains):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.domain_classifier = DomainDiscriminator(featurizer.d_out, n_domains)
        self.gradient_reverse_layer = GradientReverseLayer()

    def forward(self, input):
        features = self.featurizer(input)
        y_pred = self.classifier(features)
        features = self.gradient_reverse_layer(features)
        domains_pred = self.domain_classifier(features)
        return y_pred, domains_pred

    def get_parameters(self, featurizer_lr=1.0, classifier_lr=1.0, discriminator_lr=1.0) -> List[Dict]:
        """
        Adapted from https://github.com/thuml/Transfer-Learning-Library

        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default. For our implementation, we allow the learning
        # rates to be passed in separately for featurizer and classifier.
        params = [
            {"params": self.featurizer.parameters(), "lr": featurizer_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
        ]
        return params + self.domain_classifier.get_parameters(discriminator_lr)
