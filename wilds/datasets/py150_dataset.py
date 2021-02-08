from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch
import json
import gc

from wilds.datasets.wilds_dataset import WILDSDataset
from transformers import GPT2Tokenizer

class Py150Dataset(WILDSDataset):
    """
        The Py150 dataset.
        This is a modified version of the original Py150 dataset.
        Input (x):
            A Python code snippet (a sequence of tokens)
        Label (y):
            A sequence of next tokens (shifted x)
        Metadata:
            Each image is annotated with the original GitHub repo id and file name
        Website:
            https://www.sri.inf.ethz.ch/py150
            https://github.com/microsoft/CodeXGLUE
        Original publication:
            @article{raychev2016probabilistic,
              title={Probabilistic model for code with decision trees},
              author={Raychev, Veselin and Bielik, Pavol and Vechev, Martin},
              journal={ACM SIGPLAN Notices},
              year={2016},
            }
            @article{CodeXGLUE,
              title={CodeXGLUE: A Benchmark Dataset and Open Challenge for Code Intelligence},
              year={2020},
            }
        License:
            This dataset is distributed under the MIT license.
        """

    def __init__(self, root_dir='data', download=False, split_scheme='official'):

        self._dataset_name = 'py150'
        self._version = '1.0'
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # path
        self._download_url = 'https://worksheets.codalab.org/rest/bundles/0x45343bd9e1c64acfbcb4a22a76302994/contents/blob/'
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load data
        df = self._load_all_data()

        # Splits
        data = {}
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'IDval': 3, 'IDtest': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD)',
                                'test': 'Test (OOD)', 'IDval': 'Validation (ID)',
                                'IDtest': 'Test (ID)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df['split_id'].values

        # Input
        self._input_array = torch.tensor(list(df['input'].apply(lambda x: x[:-1]).values)) #[n_samples, seqlen-1]

        # Labels
        name = 'microsoft/CodeGPT-small-py'
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        self._n_classes = len(tokenizer)
        self._y_array = torch.tensor(list(df['input'].apply(lambda x: x[1:]).values))
        self._y_size = None

        _repo = torch.tensor(df['repo'].values).reshape(-1,1)  #[n_samples, 1]
        _mask = torch.tensor(list(df['mask'].apply(lambda x: x[1:]).values)) #[n_samples, seqlen-1]
        self._metadata_array = _repo
        self._metadata_fields = ['repo']

        self._y_array = self._y_array * _mask + torch.full(self._y_array.size(), -100) * (1-_mask)

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata):
        #y_pred: [n_samples, seqlen-1]
        #y_true: [n_samples, seqlen-1]
        mask = (y_true != -100).long()
        assert y_pred.size() == mask.size() == y_true.size(), (y_pred.size(), y_true.size(), mask.size())
        acc = ((y_pred==y_true)*mask).float().sum() / (mask.float().sum() +1e-8)

        results = {'acc': acc}
        results_str = f"Average acc: {results['acc']:.3f}\n"
        return results, results_str

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        return self._input_array[idx]


    def _load_all_data(self):
        def fname2repo_id(fname, repo_name2id):
            return repo_name2id['/'.join(fname.split('/')[:2])]

        _df = pd.read_csv(self._data_dir/'metadata/repo_file_names/repo_ids.csv')
        repo_name2id = {repo_name: id for id, repo_name in zip(_df.id, _df.repo_name)}

        dfs = []
        pad_token_id = 1
        for type in ['train', 'IDval', 'OODval', 'IDtest', 'OODtest']:
            inputs = json.load(open(self._data_dir/f'processed/{type}_input.json'))
            fnames = open(self._data_dir/f'metadata/repo_file_names/{type}.txt').readlines()
            repo_ids = [fname2repo_id(fname, repo_name2id) for fname in fnames]
            splits   = [type.replace('OOD','')] * len(inputs)
            if type == 'train':
                masks = (np.array(inputs) != pad_token_id).astype(int).tolist()
            else:
                masks = json.load(open(self._data_dir/f'processed/{type}_input_mask.json'))
            assert len(repo_ids) == len(inputs) == len(masks)

            _df  = pd.DataFrame({'input':inputs, 'mask': masks, 'repo': repo_ids, 'split': splits})
            dfs.append(_df)

        return pd.concat(dfs)
