# SwAV pre-training

This folder is contains a lightly modified version of the SwAV code from https://github.com/facebookresearch/swav, licensed under CC BY-NC 4.0.

If you use this algorithm, please cite the original source:
```
@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

SwAV requires installation of the NVIDIA Apex library for mixed-precision training. Each Apex installation has a specific CUDA extension--more information can be found in the "requirements" section of the original SwAV repository's README: ([link](https://github.com/facebookresearch/swav)).

## Changes
We made the following changes to the SwAV repository to interface with the WILDS code.

### `multicropdataset.py`
- Added a new dataset class, CustomSplitMultiCropDataset, to accommodate WILDS data loaders, allowing SwAV to train on multiple datasets at once.
### Model building code
- Pulled the changes from standard ResNets to SwAV-compatible ResNets into a new file (`model.py`), allowing to incorporate WILDS-Unlabeled architectures, including ResNets and DenseNets.
### `main_swav.py`
- Edited data loading and model building code to be compatible with the 2 changes noted above.

## Pre-training on WILDS

To run SwAV pre-training on the WILDS datasets with the default hyperparameters used in the [paper](https://arxiv.org/abs/2112.05090),
simply run:

```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=<Number of GPUs> main_swav.py --dataset <WILDS dataset> --root_dir <Path to dataset>
```

We support SwAV pre-training on the following datasets:

- `camelyon17`
- `iwildcam`
- `fmow`
- `poverty`
- `domainnet`