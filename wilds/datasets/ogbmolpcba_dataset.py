import os
import torch
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.utils.url import download_url
import torch_geometric
if torch_geometric.__version__ >= '2.0.0':
    from torch_geometric.loader.dataloader import Collater as PyGCollater
else:
    from torch_geometric.data.dataloader import Collater as PyGCollater

class OGBPCBADataset(WILDSDataset):
    """
    The OGB-molpcba dataset.
    This dataset is directly adopted from Open Graph Benchmark, and originally curated by MoleculeNet.

    Supported `split_scheme`:
        - 'official' or 'scaffold', which are equivalent

    Input (x):
        Molecular graphs represented as Pytorch Geometric data objects

    Label (y):
        y represents 128-class binary labels.

    Metadata:
        - scaffold
            Each molecule is annotated with the scaffold ID that the molecule is assigned to.

    Website:
        https://ogb.stanford.edu/docs/graphprop/#ogbg-mol

    Original publication:
        @article{hu2020ogb,
            title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
            author={W. {Hu}, M. {Fey}, M. {Zitnik}, Y. {Dong}, H. {Ren}, B. {Liu}, M. {Catasta}, J. {Leskovec}},
            journal={arXiv preprint arXiv:2005.00687},
            year={2020}
        }

        @article{wu2018moleculenet,
            title={MoleculeNet: a benchmark for molecular machine learning},
            author={Z. {Wu}, B. {Ramsundar}, E. V {Feinberg}, J. {Gomes}, C. {Geniesse}, A. S {Pappu}, K. {Leswing}, V. {Pande}},
            journal={Chemical science},
            volume={9},
            number={2},
            pages={513--530},
            year={2018},
            publisher={Royal Society of Chemistry}
        }

    License:
        This dataset is distributed under the MIT license.
        https://github.com/snap-stanford/ogb/blob/master/LICENSE
    """

    _dataset_name = 'ogb-molpcba'
    _versions_dict = {
        '1.0': {
            'download_url': None,
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        if version is not None:
            raise ValueError('Versioning for OGB-MolPCBA is handled through the OGB package. Please set version=none.')
        # internally call ogb package
        self.ogb_dataset = PygGraphPropPredDataset(name = 'ogbg-molpcba', root = root_dir)

        # set variables
        self._data_dir = self.ogb_dataset.root
        if split_scheme=='official':
            split_scheme = 'scaffold'
        self._split_scheme = split_scheme
        self._y_type = 'float' # although the task is binary classification, the prediction target contains nan value, thus we need float
        self._y_size = self.ogb_dataset.num_tasks
        self._n_classes = self.ogb_dataset.__num_classes__

        self._split_array = torch.zeros(len(self.ogb_dataset)).long()
        split_idx  = self.ogb_dataset.get_idx_split()
        self._split_array[split_idx['train']] = 0
        self._split_array[split_idx['valid']] = 1
        self._split_array[split_idx['test']] = 2

        self._y_array = self.ogb_dataset.data.y

        self._metadata_fields = ['scaffold']

        metadata_file_path = os.path.join(self.ogb_dataset.root, 'raw', 'scaffold_group.npy')
        if not os.path.exists(metadata_file_path):
            download_url('https://snap.stanford.edu/ogb/data/misc/ogbg_molpcba/scaffold_group.npy', os.path.join(self.ogb_dataset.root, 'raw'))
        self._metadata_array = torch.from_numpy(np.load(metadata_file_path)).reshape(-1,1).long()

        if torch_geometric.__version__ >= '1.7.0':
            self._collate = PyGCollater(follow_batch=[], exclude_keys=[])
        else:
            self._collate = PyGCollater(follow_batch=[])

        self._metric = Evaluator('ogbg-molpcba')

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self.ogb_dataset[int(idx)]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (FloatTensor): Binary logits from a model
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels.
                                        Only None is supported because OGB Evaluators accept binary logits
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        assert prediction_fn is None, "OGBPCBADataset.eval() does not support prediction_fn. Only binary logits accepted"
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        results = self._metric.eval(input_dict)

        return results, f"Average precision: {results['ap']:.3f}\n"
