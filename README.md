<p align='center'>
  <img width='40%' src='https://wilds.stanford.edu/WILDS_cropped.png' />
</p>

--------------------------------------------------------------------------------

[![PyPI](https://img.shields.io/pypi/v/wilds)](https://pypi.org/project/wilds/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/p-lambda/wilds/blob/master/LICENSE)

## Overview
WILDS is a benchmark of in-the-wild distribution shifts spanning diverse data modalities and applications, from tumor identification to wildlife monitoring to poverty mapping.

The WILDS package contains:
1. Data loaders that automatically handle data downloading, processing, and splitting, and
2. Dataset evaluators that standardize model evaluation for each dataset.

In addition, the example scripts contain default models, optimizers, schedulers, and training/evaluation code. 
New algorithms can be easily added and run on all of the WILDS datasets.

For more information, please visit [our website](https://wilds.stanford.edu) or read the main WILDS paper ([1](https://arxiv.org/abs/2012.07421)) and its follow-up integrating unlabeled data ([2](https://arxiv.org/abs/2112.05090)).
For questions and feedback, please post on the [discussion board](https://github.com/p-lambda/wilds/discussions).

## Installation

We recommend using pip to install WILDS:
```bash
pip install wilds
```

If you have already installed it, please check that you have the latest version:
```bash
python -c "import wilds; print(wilds.__version__)"
# This should print "2.0.0". If it doesn't, update by running:
pip install -U wilds
```

If you plan to edit or contribute to WILDS, you should install from source:
```bash
git clone git@github.com:p-lambda/wilds.git
cd wilds
pip install -e .
```

In `examples/`, we provide a set of scripts that can be used to train models on the WILDS datasets. These scripts were also used to benchmark baselines in our papers [[1](https://arxiv.org/abs/2012.07421), [2](https://arxiv.org/abs/2112.05090)]. 
These scripts are not part of the installed WILDS package. To use them, you should install from source, as described above.

### Requirements
The WILDS package depends on the following requirements:

- numpy>=1.19.1
- ogb>=1.2.6
- outdated>=0.2.0
- pandas>=1.1.0
- pillow>=7.2.0
- pytz>=2020.4
- torch>=1.7.0
- torch-scatter>=2.0.5
- torch-geometric>=2.0.1
- torchvision>=0.8.2
- tqdm>=4.53.0
- scikit-learn>=0.20.0
- scipy>=1.5.4

Running `pip install wilds` or `pip install -e .` will automatically check for and install all of these requirements
**except for the `torch-scatter` and `torch-geometric` packages**, which require a
[quick manual install](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries).

### Example script requirements
To run the example scripts, you will also need to install these additional dependencies:

- transformers>=3.5.0
- SwAV requires [Apex](https://github.com/NVIDIA/apex).
  To install Apex, please follow the [README from the official SwAV repository](https://github.com/facebookresearch/swav#requirements).
- Our code supports the optional use of [Weights & Biases](https://wandb.ai/site) to track and monitor experiments.
  To install the Weights and Biases Python package, run `pip install wandb`.

All baseline experiments in the paper were run on Python 3.8.5 and CUDA 10.1.

## Datasets
WILDS currently includes 10 datasets, which we've briefly listed below. For full dataset descriptions, please see our papers ([1](https://arxiv.org/abs/2012.07421), [2](https://arxiv.org/abs/2112.05090)).

| Dataset       | Modality | Labeled splits                    | Unlabeled splits                                                |
| ------------- | -------- | --------------------------------- | --------------------------------------------------------------- |
| iwildcam      | Image    | train, val, test, id_val, id_test | extra_unlabeled                                                 |
| camelyon17    | Image    | train, val, test, id_val          | train_unlabeled, val_unlabeled, test_unlabeled                  |
| rxrx1         | Image    | train, val, test, id_test         | -                                                               |
| ogb-molpcba   | Graph    | train, val, test                  | train_unlabeled, val_unlabeled, test_unlabeled                  |
| globalwheat   | Image    | train, val, test, id_val, id_test | train_unlabeled, val_unlabeled, test_unlabeled, extra_unlabeled |
| civilcomments | Text     | train, val, test                  | extra_unlabeled                                                 |
| fmow          | Image    | train, val, test, id_val, id_test | train_unlabeled, val_unlabeled, test_unlabeled                  |
| poverty       | Image    | train, val, test, id_val, id_test | train_unlabeled, val_unlabeled, test_unlabeled                  |
| amazon        | Text     | train, val, test, id_val, id_test | val_unlabeled, test_unlabeled, extra_unlabeled                  |
| py150         | Text     | train, val, test, id_val, id_test | -                                                               |

## Using the WILDS package
### Data

The WILDS package provides a simple, standardized interface for all datasets in the benchmark.
This short Python snippet covers all of the steps of getting started with a WILDS dataset, including dataset download and initialization, accessing various splits, and preparing a user-customizable data loader.
We discuss data loading in more detail in [#Data loading](#data-loading).

```py
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="iwildcam", download=True)

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)

# Prepare the standard data loader
train_loader = get_train_loader("standard", train_data, batch_size=16)

# (Optional) Load unlabeled data
dataset = get_dataset(dataset="iwildcam", download=True, unlabeled=True)
unlabeled_data = dataset.get_subset(
    "test_unlabeled",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)

# Train loop
for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_loader):
    x, y, metadata = labeled_batch
    unlabeled_x, unlabeled_metadata = unlabeled_batch
    ...
```

The `metadata` contains information like the domain identity, e.g., which camera a photo was taken from, or which hospital the patient's data came from, etc., as well as other metadata.

### Domain information
To allow algorithms to leverage domain annotations as well as other groupings over the available metadata, the WILDS package provides `Grouper` objects.
These `Grouper` objects are helper objects that extract group annotations from metadata, allowing users to specify the grouping scheme in a flexible fashion.
They are used to initialize group-aware data loaders (as discussed in [#Data loading](#data-loading)) and to implement algorithms that rely on domain annotations (e.g., Group DRO).
In the following code snippet, we initialize and use a `Grouper` that extracts the domain annotations on the iWildCam dataset, where the domain is location.

```py
from wilds.common.grouper import CombinatorialGrouper

# Initialize grouper, which extracts domain information
# In this example, we form domains based on location
grouper = CombinatorialGrouper(dataset, ['location'])

# Train loop
for x, y_true, metadata in train_loader:
    z = grouper.metadata_to_group(metadata)
    ...
```

### Data loading

For training, the WILDS package provides two types of data loaders.
The standard data loader shuffles examples in the training set, and is used for the standard approach of empirical risk minimization (ERM), where we minimize the average loss.
```py
from wilds.common.data_loaders import get_train_loader

# Prepare the standard data loader
train_loader = get_train_loader('standard', train_data, batch_size=16)
```

To support other algorithms that rely on specific data loading schemes, we also provide the group data loader.
In each minibatch, the group loader first samples a specified number of groups, and then samples a fixed number of examples from each of those groups.
(By default, the groups are sampled uniformly at random, which upweights minority groups as a result. This can be toggled by the `uniform_over_groups` parameter.)
We initialize group loaders as follows, using `Grouper` that specifies the grouping scheme.

```py
# Prepare a group data loader that samples from user-specified groups
train_loader = get_train_loader(
    "group", train_data, grouper=grouper, n_groups_per_batch=2, batch_size=16
)

```

Lastly, we also provide a data loader for evaluation, which loads examples without shuffling (unlike the training loaders).

```py
from wilds.common.data_loaders import get_eval_loader

# Get the test set
test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

# Prepare the evaluation data loader
test_loader = get_eval_loader("standard", test_data, batch_size=16)

```

### Evaluators

The WILDS package standardizes and automates evaluation for each dataset.
Invoking the `eval` method of each dataset yields all metrics reported in the paper and on the leaderboard.

```py
from wilds.common.data_loaders import get_eval_loader

# Get the test set
test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

# Prepare the data loader
test_loader = get_eval_loader("standard", test_data, batch_size=16)

# Get predictions for the full test set
for x, y_true, metadata in test_loader:
    y_pred = model(x)
    # Accumulate y_true, y_pred, metadata

# Evaluate
dataset.eval(all_y_pred, all_y_true, all_metadata)
# {'recall_macro_all': 0.66, ...}
```
Most `eval` methods take in predicted labels for `all_y_pred` by default, but the default inputs vary across datasets and are documented in the `eval` docstrings of the corresponding dataset class.

## Using the example scripts
In `examples/`, we provide a set of scripts that can be used to train models on the WILDS datasets. 

```bash
python examples/run_expt.py --dataset iwildcam --algorithm ERM --root_dir data
python examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data
python examples/run_expt.py --dataset fmow --algorithm DANN --unlabeled_split test_unlabeled --root_dir data
```

The scripts are configured to use the default models and reasonable hyperparameters. For exact hyperparameter settings used in our papers, please see [our CodaLab executable paper](https://wilds.stanford.edu/codalab).

### Downloading and training on the WILDS datasets
The first time you run these scripts, you might need to download the datasets. You can do so with the `--download` argument, for example:
```python
# downloads (labeled) dataset
python examples/run_expt.py --dataset globalwheat --algorithm groupDRO --root_dir data --download

# additionally downloads all unlabeled data
python examples/run_expt.py --dataset globalwheat --algorithm groupDRO --root_dir data --download  --unlabeled_split [...]
```
Note that downloading the large amount of unlabeled data is optional; unlabeled data will only be downloaded if some `--unlabeled_split` is set. (It does not matter which `--unlabeled_split` is set; all unlabeled data will be downloaded together.)

Alternatively, you can use the standalone `wilds/download_datasets.py` script to download the datasets, for example:

```bash
# downloads (labeled) data
python wilds/download_datasets.py --root_dir data

# downloads (unlabeled) data
python wilds/download_datasets.py --root_dir data --unlabeled
```

This will download all datasets to the specified `data` folder. You can also use the `--datasets` argument to download particular datasets.

These are the sizes of each of our datasets, as well as their approximate time taken to train and evaluate the default model for a single ERM run using a NVIDIA V100 GPU.

| Dataset command | Modality | Download size (GB) | Size on disk (GB) | Train+eval time (Hours) |
| --------------- | -------- | ------------------ | ----------------- | ----------------------- |
| iwildcam        | Image    | 11                 | 25                | 7                       |
| camelyon17      | Image    | 10                 | 15                | 2                       |
| rxrx1           | Image    | 7                  | 7                 | 11                      |
| ogb-molpcba     | Graph    | 0.04               | 2                 | 15                      |
| globalwheat     | Image    | 10                 | 10                | 2                       |
| civilcomments   | Text     | 0.1                | 0.3               | 4.5                     |
| fmow            | Image    | 50                 | 55                | 6                       |
| poverty         | Image    | 12                 | 14                | 5                       |
| amazon          | Text     | 7                  | 7                 | 5                       |
| py150           | Text     | 0.1                | 0.8               | 9.5                     |

The following are the sizes of the unlabeled data bundles:

| Dataset command | Modality | Download size (GB) | Size on disk (GB) |
| --------------- | -------- | ------------------ | ----------------- |
| iwildcam        | Image    | 41                 | 41                |
| camelyon17      | Image    | 69.4               | 96                |
| ogb-molpcba     | Graph    | 1.2                | 21                |
| globalwheat     | Image    | 103                | 108               |
| civilcomments   | Text     | 0.3                | 0.6               |
| fmow\*          | Image    | 50                 | 55                |
| poverty         | Image    | 172                | 184               |
| amazon\*        | Text     | 7                  | 7                 |

<sup> \* These unlabeled datasets are downloaded simultaneously with the labeled data and do not need to be downloaded separately. </sup>

While the `camelyon17` dataset is small and fast to train on, we advise against using it as the only dataset to prototype methods on, as the test performance of models trained on this dataset tend to exhibit a large degree of variability over random seeds.

The image datasets (`iwildcam`, `camelyon17`, `rxrx1`, `globalwheat`, `fmow`, and `poverty`) tend to have high disk I/O usage. If training time is much slower for you than the approximate times listed above, consider checking if I/O is a bottleneck (e.g., by moving to a local disk if you are using a network drive, or by increasing the number of data loader workers). To speed up training, you could also disable evaluation at each epoch or for all splits by toggling `--evaluate_all_splits` and related arguments.

### Algorithms
In the `examples/algorithms` folder, we provide implementations of the adaptation algorithms benchmarked in our papers ([1](https://arxiv.org/abs/2012.07421), [2](https://arxiv.org/abs/2112.05090)).
All algorithms train on labeled data from a WILDS dataset's `train` split.
Some algorithms are designed to also leverage unlabeled data. To load unlabeled data, specify an `--unlabeled_split` when running.

In addition to shared hyperparameters such as `lr`, `weight_decay`, `batch_size`, and `unlabeled_batch_size`, the scripts also take in command line arguments for algorithm-specific hyperparameters.

| Algorithm command                                | Hyperparameters                                                                            | Notes                             | See WILDS paper                                                                |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------------------------ |
| ERM                                              | -                                                                                          | Only uses labeled data            | ([1](https://arxiv.org/abs/2012.07421), [2](https://arxiv.org/abs/2112.05090)) |
| [groupDRO](https://arxiv.org/abs/1911.08731)     | `group_dro_step_size`                                                                      | Only uses labeled data            | ([1](https://arxiv.org/abs/2012.07421))                                        |
| [deepCORAL](https://arxiv.org/abs/1511.05547)    | `coral_penalty_weight`                                                                     | Can optionally use unlabeled data | ([1](https://arxiv.org/abs/2012.07421), [2](https://arxiv.org/abs/2112.05090)) |
| [IRM](https://arxiv.org/abs/1907.02893)          | `irm_lambda`, `irm_penalty_anneal_iters`                                                   | Only uses labeled data            | ([1](https://arxiv.org/abs/2012.07421))                                        |
| [DANN](https://arxiv.org/abs/1505.07818)         | `dann_penalty_weight`, `dann_classifier_lr`, `dann_featurizer_lr`, `dann_discriminator_lr` | Can use unlabeled data            | ([2](https://arxiv.org/abs/2112.05090))                                        |
| [AFN](https://arxiv.org/abs/1811.07456)          | `afn_penalty_weight`, `safn_delta_r`, `hafn_r`                                             | Designed to use unlabeled data    | ([2](https://arxiv.org/abs/2112.05090))                                        |
| [FixMatch](https://arxiv.org/abs/2001.07685)     | `self_training_lambda`, `self_training_threshold`                                          | Designed to use unlabeled data    | ([2](https://arxiv.org/abs/2112.05090))                                        |
| PseudoLabel                                      | `self_training_lambda`, `self_training_threshold`, `pseudolabel_T2`                        | Designed to use unlabeled data    | ([2](https://arxiv.org/abs/2112.05090))                                        |
| [NoisyStudent](https://arxiv.org/abs/1911.04252) | `soft_pseudolabels`, `noisystudent_dropout_rate`                                           | Designed to use unlabeled data    | ([2](https://arxiv.org/abs/2112.05090))                                        |

The repository is set up to facilitate general-purpose algorithm development: new algorithms can be added to `examples/algorithms` and then run on all of the WILDS datasets using the default models.

### Evaluating trained models
We also provide an evaluation script that aggregates prediction CSV files for different replicates and reports on their combined evaluation. To use this, run:

```bash
python examples/evaluate.py <predictions_dir> <output_dir> --root-dir <root_dir>
```

where `<predictions_dir>` is the path to your predictions directory, `<output_dir>` is where the results JSON will be writte, and `<root_dir>` is the dataset root directory.
The predictions directory should have a subdirectory for each dataset
(e.g. `iwildcam`) containing prediction CSV files to evaluate; see our [submission guidelines](https://wilds.stanford.edu/submit/) for the format.
The evaluation script will skip over any datasets that has missing prediction files.
Any dataset not in `<root_dir>` will be downloaded to `<root_dir>`.

### Reproducibility
We have an [executable version](https://wilds.stanford.edu/codalab) of our paper on CodaLab that contains the exact commands, code, and data for the experiments reported in our paper, which rely on these scripts. Trained model weights for all datasets can also be found there.
All configurations and hyperparameters can also be found in the `examples/configs` folder of this repo, and dataset-specific parameters are in `examples/configs/datasets.py`.

## Leaderboard
If you are developing new training algorithms and/or models on WILDS, please consider submitting them to our [public leaderboard](https://wilds.stanford.edu/leaderboard/).

## Citing WILDS ([Bibtex](https://wilds.stanford.edu/assets/files/wilds_bib.txt))
If you use WILDS datasets in your work, please cite our paper:

1. [**WILDS: A Benchmark of in-the-Wild Distribution Shifts.**](https://arxiv.org/abs/2012.07421) Pang Wei Koh*, Shiori Sagawa*, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. ICML 2021.

If you use unlabeled data from the WILDS datasets, please also cite:

2. [**Extending the WILDS Benchmark for Unsupervised Adaptation.**](https://arxiv.org/abs/2112.05090) Shiori Sagawa*, Pang Wei Koh*, Tony Lee*, Irena Gao*, Sang Michael Xie, Kendrick Shen, Ananya Kumar, Weihua Hu, Michihiro Yasunaga, Henrik Marklund, Sara Beery, Etienne David, Ian Stavness, Wei Guo, Jure Leskovec, Kate Saenko, Tatsunori Hashimoto, Sergey Levine, Chelsea Finn, and Percy Liang. NeurIPS 2021 Workshop on Distribution Shifts.

In addition, please cite the original papers that introduced the datasets, as listed on the [datasets page](https://wilds.stanford.edu/datasets/).

## Acknowledgements
The design of the WILDS benchmark was inspired by the [Open Graph Benchmark](https://ogb.stanford.edu/), and we are grateful to the Open Graph Benchmark team for their advice and help in setting up WILDS.
