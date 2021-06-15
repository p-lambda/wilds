from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]

class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library

    coeff is the same as lambda (domain adaptation parameter) in the original paper
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
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

    The forward and backward behaviours are:

    .. math::
        \mathcal{R}(x) = x,

        \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

    :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

    .. math::
        \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

    where :math:`i` is the iteration step.

    Args:
        alpha (float, optional): :math:`α`. Default: 1.0
        lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
        hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
        max_iters (int, optional): :math:`N`. Default: 1000
        auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
          Otherwise use function `step` to increase :math:`i`. Default: True
    """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = True):
        super(GradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class DomainAdversarialNetwork(nn.Module):
    def __init__(self, featurizer, classifier, n_domains):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.domain_classifier= DomainDiscriminator(featurizer.d_out, n_domains)
        self.gradient_reverse_layer = GradientReverseLayer()

    def forward(self, input):
        features = self.featurizer(input)
        y_pred = self.classifier(features)
        features = self.gradient_reverse_layer(features)
        domains_pred = self.domain_classifier(features)
        return y_pred, domains_pred

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """
        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # From TLL, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default.
        params = [
            {"params": self.featurizer.parameters(), "lr": 0.1 * base_lr},
            {"params": self.classifier.parameters(), "lr": 1.0 * base_lr},
        ]
        return params + self.domain_classifier.get_parameters()
