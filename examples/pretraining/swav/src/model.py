#
# This file defines the SwAVModel class, a wrapper around WILDS-Unlabeled architectures
# that implements the changes necessary to make the networks compatible with SwAV
# training (e.g. prototypes, projection head, etc.). Currently, the supported architectures
# are ResNets and DenseNets.
#

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
import examples.models.resnet_multispectral as resnet_ms

class SwAVModel(nn.Module):
    def __init__(
        self,
        base_model,
        normalize=False,
        output_dim=0,
        hidden_mlp=0,
        nmb_prototypes=0,
    ):
        super(SwAVModel, self).__init__()

        self.base_model = base_model # base CNN architecture
        self.l2norm = normalize # whether to normalize output features

        # projection head
        last_dim = base_model.d_out # output dimensionality of final featurizer layer
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(last_dim, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(last_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = F.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.base_model(
                torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out
