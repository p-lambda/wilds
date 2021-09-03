### Changes to `multicropdataset.py`
- Added a new dataset class, CustomSplitMultiCropDataset, to accomodate WILDS data loaders, allowing SwAV to train on multiple datasets at once.
### Changes to model building code
- Pulled the changes from standard ResNets to SwAV-compatible ResNets into a new file (`model.py`), allowing to incorporate WILDS-Unlabeled architectures, including ResNets and DenseNets.
### Changes to `main_swav.py`
- Edited data loading and model building code to be compatible with the 2 changes noted above.

## Notes
- SwAV requires installation of the NVIDIA Apex library for mixed-precision training. Each Apex installation has a specific CUDA extension--more information can be found in the "requirements" section of the original SwAV repository's README: ([link](https://github.com/facebookresearch/swav)).
