import os
import torch
import numpy as np

from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils.url import download_url
import torch_geometric
if torch_geometric.__version__ >= '2.0.0':
    from torch_geometric.loader.dataloader import Collater as PyGCollater
else:
    from torch_geometric.data.dataloader import Collater as PyGCollater

class OGBPCBAUnlabeledDataset(WILDSUnlabeledDataset):
    """
    Unlabeled dataset for OGB-molpcba. There are 5 million unlabeled molecules randomly sampled from the entire PubChem database.

    Input (x):
        Molecular graphs represented as Pytorch Geometric data objects

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

    License:
        This dataset is distributed under the MIT license.
        https://github.com/snap-stanford/ogb/blob/master/LICENSE
    """

    _dataset_name = 'ogb-molpcba_unlabeled'
    _versions_dict = {
        '1.0': {
            'download_url': None,
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        if version is not None:
            raise ValueError('Versioning for Unlabeled MolPCBA is handled through the OGB package. Please set version=none.')
        # internally call ogb package
        
        ### Setting up meta-information for the dataset
        meta_dict = {}
        meta_dict['dir_path'] = os.path.join(root_dir, 'molpcba_unlabeled')
        meta_dict['url'] = 'http://snap.stanford.edu/ogb/data/wilds/molpcba_unlabeled.zip'
        meta_dict['num tasks'] = 0
        meta_dict['eval metric'] = None
        meta_dict['download_name'] = 'molpcba_unlabeled'
        meta_dict['version'] = 1
        meta_dict['add_inverse_edge'] = 'False'
        meta_dict['data type'] = 'mol'
        meta_dict['has_node_attr'] = 'True'
        meta_dict['has_edge_attr'] = 'True'
        meta_dict['task type'] = 'classification'
        meta_dict['num classes'] = -1
        meta_dict['split'] = 'scaffold'
        meta_dict['additional node files'] = 'None'
        meta_dict['additional edge files'] = 'None'
        meta_dict['binary'] = 'True'

        self.ogb_dataset = PygGraphPropPredDataset(name = 'molpcba_unlabeled', root = root_dir, meta_dict = meta_dict)
        self.ogb_dataset.data.y = None

        # set variables
        self._data_dir = self.ogb_dataset.root
        if split_scheme=='official':
            split_scheme = 'scaffold'
        self._split_scheme = split_scheme

        self._split_array = torch.zeros(len(self.ogb_dataset)).long()
        split_idx  = self.ogb_dataset.get_idx_split()
        self._split_array[split_idx['train']] = 10
        self._split_array[split_idx['valid']] = 11
        self._split_array[split_idx['test']] = 12

        self._metadata_fields = ['scaffold']

        metadata_file_path = os.path.join(self.ogb_dataset.root, 'processed', 'group_assignment.npy')
        self._metadata_array = torch.from_numpy(np.load(metadata_file_path)).reshape(-1,1).long()

        if torch_geometric.__version__ >= '1.7.0':
            self._collate = PyGCollater(follow_batch=[], exclude_keys=[])
        else:
            self._collate = PyGCollater(follow_batch=[])

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self.ogb_dataset[int(idx)]
