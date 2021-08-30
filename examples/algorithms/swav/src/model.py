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
        eval_mode=False
    ):
        super(SwAVModel, self).__init__()

        self.base_model = base_model # base CNN architecture
        self.eval_mode = eval_mode # whether to compute prototypes / codes
        self.l2norm = normalize # whether to normalize output features

        # projection head
        last_dim = get_final_layer_dim(base_model)
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
        
        self.forward_backbone = get_forward_backbone(base_model)

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
            _out = self.forward_backbone(
                self.base_model,
                torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True),
                self.eval_mode)
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

def get_final_layer_dim(model):
    if isinstance(model, models.ResNet) or isinstance(model, resnet_ms.ResNet):
        last_layer_name = 'fc'
    elif isinstance(model, models.DenseNet):
        last_layer_name = 'classifier'
    else:
        raise NotImplementedError('Supported model classes are ResNets and DenseNets.')
    return getattr(model, last_layer_name).in_features

def get_forward_backbone(model):
    if isinstance(model, models.ResNet):
        def func(model, x, eval_mode):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            if eval_mode:
                return x
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        return func
    if isinstance(model, models.DenseNet):
        def func(model, x, eval_mode):
            features = model.features(x)
            out = F.relu(features, inplace=True)
            if eval_mode:
                return features
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            return out
        return func
    if isinstance(model, resnet_ms.ResNet):
        raise NotImplementedError()
    raise NotImplementedError('Supported model classes are ResNets and DenseNets.')
