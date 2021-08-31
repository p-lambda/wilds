### Changes to `multicropdataset.py`
- Added a new dataset class to accomodate WILDS data loaders
### Changes to model building code
- Moved the edits from standard ResNets to SwAV-compatible ResNets into a new file (`model.py`), allowing to incorporate more model architectures
### Changes to `main_swav.py`
- Edited data loading and model building code to be compatible with other changes
