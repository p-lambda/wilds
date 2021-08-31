import argparse
import os
import sys
import copy
import pickle

import torch
import torchvision.transforms as transforms
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from src.model import SwAVModel
from src.utils import ParseKwargs
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from examples.models.initializer import initialize_model

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
)
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])

parser = argparse.ArgumentParser(description='Extract features from model.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# training args
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
# swav checkpoint args
parser.add_argument('--run_dir', type=str, required=True,
                    help='The (outer) SwAV run directory to use.')
parser.add_argument('--ckpt_epoch', type=int, required=True,
                    help='The epoch of the checkpoint in the checkpoints/ folder.')
# dataset args
parser.add_argument('-d', '--dataset', required=True, choices=wilds.unlabeled_datasets)
parser.add_argument('--root_dir', required=True,
                    help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={})
# model args
parser.add_argument("--model", type=str, help="convnet architecture. If not set, uses default model specified in WILDS.")
parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for model initialization passed as key1=value1 key2=value2')

def get_model(args):
    d_out = 1 # this can be arbitrary; final layer is discarded for SwAVModel
    base_model = initialize_model(args, d_out, **args.model_kwargs)
    model = SwAVModel(base_model, output_dim=0, eval_mode=True)
    ckpt = torch.load(
        os.path.join(args.run_dir, 'checkpoints', f'ckp-{args.ckpt_epoch}.pth'),
        map_location='cpu'
    )
    state_dict = ckpt['state_dict']
    state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }
    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    assert len(missing_keys) == 0, f'Was unable to match keys: {",".join(missing_keys)}'
    return model.eval().cuda()

def get_data(args):
    dataset = wilds.get_dataset(
        dataset=args.dataset,
        root_dir=args.root_dir,
        download=True,
        **args.dataset_kwargs
    )
    train_data = dataset.get_subset('train', transform=TRANSFORM)
    test_data = dataset.get_subset('test', transform=TRANSFORM)
    train_loader = get_train_loader('standard', train_data, batch_size=args.batch_size)
    test_loader = get_eval_loader('standard', test_data, batch_size=args.batch_size)
    return train_loader, test_loader

def get_feats(model, data):
    feats = []
    for loader in data:
        features_list, labels_list = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.cuda()
                features = model(x)
                features_list.append(features.detach().cpu().numpy())
                labels_list.append(y.detach().numpy())
        features = np.squeeze(np.concatenate(features_list))
        labels = np.concatenate(labels_list)
        feats.append([features, labels])
    return feats

def main():
    args = parser.parse_args()
    # saving
    os.makedirs(os.path.join(args.run_dir, 'finetuning'), exist_ok=True)
    file_prefix = f'features_and_labels_{args.ckpt_epoch}'
    file_path = os.path.join(args.run_dir, 'finetuning', f'{file_prefix}.pickle')

    model = get_model(args)
    data = get_data(args)
    feats = get_feats(args, model, data)
    pickle.dump(feats, open(file_path, 'wb'))

if __name__ == "__main__":
    main()
