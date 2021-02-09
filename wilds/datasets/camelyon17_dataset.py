import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class Camelyon17Dataset(WILDSDataset):
    """
    The CAMELYON17-wilds histopathology dataset.
    This is a modified version of the original CAMELYON17 dataset.

    Supported `split_scheme`:
        'official' or 'in-dist'

    Input (x):
        96x96 image patches extracted from histopathology slides.

    Label (y):
        y is binary. It is 1 if the central 32x32 region contains any tumor tissue, and 0 otherwise.

    Metadata:
        Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
        and the slide it came from (integer from 0 to 49).

    Website:
        https://camelyon17.grand-challenge.org/

    Original publication:
        @article{bandi2018detection,
          title={From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge},
          author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
          journal={IEEE transactions on medical imaging},
          volume={38},
          number={2},
          pages={550--560},
          year={2018},
          publisher={IEEE}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        self._dataset_name = 'camelyon17'
        self._version = '1.0'
        self._download_url = 'https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/'
        self._compressed_size = 10_658_709_504
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (96,96)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['tumor'].values)
        self._y_size = 1
        self._n_classes = 2

        # Get filenames
        self._input_array = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            self._metadata_df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]

        # Extract splits
        # Note that the hospital numbering here is different from what's in the paper,
        # where to avoid confusing readers we used a 1-indexed scheme and just labeled the test hospital as 5.
        # Here, the numbers are 0-indexed.
        test_center = 2
        val_center = 1

        self._split_dict = {
            'train': 0,
            'id_val': 1,
            'test': 2,
            'val': 3
        }
        self._split_names = {
            'train': 'Train',
            'id_val': 'Validation (ID)',
            'test': 'Test',
            'val': 'Validation (OOD)',
        }
        centers = self._metadata_df['center'].values.astype('long')
        num_centers = int(np.max(centers)) + 1
        val_center_mask = (self._metadata_df['center'] == val_center)
        test_center_mask = (self._metadata_df['center'] == test_center)
        self._metadata_df.loc[val_center_mask, 'split'] = self.split_dict['val']
        self._metadata_df.loc[test_center_mask, 'split'] = self.split_dict['test']

        self._split_scheme = split_scheme
        if self._split_scheme == 'official':
            pass
        elif self._split_scheme == 'in-dist':
            # For the in-distribution oracle,
            # we move slide 23 (corresponding to patient 042, node 3 in the original dataset)
            # from the test set to the training set
            slide_mask = (self._metadata_df['slide'] == 23)
            self._metadata_df.loc[slide_mask, 'split'] = self.split_dict['train']
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = self._metadata_df['split'].values

        self._metadata_array = torch.stack(
            (torch.LongTensor(centers),
             torch.LongTensor(self._metadata_df['slide'].values),
             self._y_array),
            dim=1)
        self._metadata_fields = ['hospital', 'slide', 'y']

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['slide'])

        self._metric = Accuracy()

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       img_filename = os.path.join(
           self.data_dir,
           self._input_array[idx])
       x = Image.open(img_filename).convert('RGB')
       return x

    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)


class EncodeTFBSDataset(WILDSDataset):
    """
    EncodeTFBS dataset
    Website: https://www.synapse.org/#!Synapse:syn6131484
    """

    def __init__(self, root_dir, download, split_scheme):
        self._dataset_name = 'encodeTFBS'
        self._download_url = 'https://worksheets.codalab.org/rest/bundles/0x8b3255e21e164cd98d3aeec09cd0bc26/contents/blob/'
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._y_size = 1
        self._n_classes = 2
        
        self._tr_chrs = ['chr2', 'chr9', 'chr11']
        self._te_chrs = ['chr1', 'chr8', 'chr21']
        self._transcription_factor = 'MAX'
        self._train_celltypes = ['H1-hESC', 'HCT116', 'HeLa-S3', 'HepG2', 'K562']
        self._val_celltype = ['A549']
        self._test_celltype = ['GM12878']
        self._all_celltypes = self._train_celltypes + self._val_celltype + self._test_celltype
        
        self._metadata_fields = ['chr', 'celltype', 'y']
        self._metadata_map = {}
        self._metadata_map['chr'] = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
        self._metadata_map['celltype'] = self._all_celltypes
        
        # Load sequence and DNase features
        sequence_filename = os.path.join(self._data_dir, 'sequence.npz')
        seq_arr = np.load(sequence_filename)
        self._seq_bp = {}
        for chrom in seq_arr:
            self._seq_bp[chrom] = seq_arr[chrom]
        
        self._dnase_allcelltypes = {}
        for ct in self._all_celltypes:
            dnase_filename = os.path.join(self._data_dir, '{}_dnase.npz'.format(ct))
            dnase_npz_file = np.load(dnase_filename)
            self._dnase_allcelltypes[ct] = {}
            for chrom in seq_bp:
                self._dnase_allcelltypes[ct][chrom] = dnase_npz_file[chrom]
        
        # Read in metadata dataframe from training+validation data
        train_chr = pd.read_csv(os.path.join(self._data_dir, 'labels/{}.train.labels.tsv.gz'.format(self._transcription_factor)), sep='\t')
        val_chr = pd.read_csv(os.path.join(self._data_dir, 'labels/{}.val.labels.tsv.gz'.format(self._transcription_factor)), sep='\t')
        training_df = train_chr[np.isin(train_chr['chr'], self._tr_chrs)]
        val_df = val_chr[np.isin(val_chr['chr'], self._te_chrs)]
        all_df = pd.concat([training_df, val_df])
        
        # Filter by start/stop coordinate if needed
        filter_msk = all_df['start'] >= 0
        filter_msk = all_df['start']%1000 == 0
        all_df = all_df[filter_msk]
        
        pd_list = []
        for ct in self._train_celltypes:
            tc_chr = all_df[['chr', 'start', 'stop', ct]]
            tc_chr.columns = ['chr', 'start', 'stop', 'y']
            tc_chr['celltype'] = ct
            pd_list.append(tc_chr)
        metadata_df = pd.concat(pd_list)
        
        # Get the y values, and remove ambiguous labels by default.
        y_array = metadata_df['y'].replace({'U': 0, 'B': 1, 'A': -1}).values
        non_ambig_mask = (y_array != -1)
        metadata_df['y'] = y_array
        self._metadata_df = metadata_df[non_ambig_mask]
        self._y_array = torch.LongTensor(y_array[non_ambig_mask])
        
        chr_ints = self._metadata_df['chr'].replace(dict( [(y, x) for x, y in enumerate(self._metadata_map['chr'])] )).values
        celltype_ints = self._metadata_df['celltype'].replace(dict( [(y, x) for x, y in enumerate(self._metadata_map['celltype'])] )).values
        self._metadata_array = torch.stack(
            (torch.LongTensor(chr_ints), 
             torch.LongTensor(celltype_ints), 
             self._y_array),
            dim=1)
        
        # Get the splits
        # TODO Extract splits as encoded in split_scheme. Hardcoded here for now.
        self._split_scheme = split_scheme
        self._split_dict = {
            'train': 0,
            'val-id': 1,
            'test': 2,
            'val-ood': 3
        }
        self._split_names = {
            'train': 'Train',
            'val-id': 'Validation (ID)',
            'test': 'Test',
            'val-ood': 'Validation (OOD)',
        }
        train_chr_mask = np.isin(self._metadata_df['chr'], self._tr_chrs)
        val_chr_mask = np.isin(self._metadata_df['chr'], self._te_chrs)
        train_celltype_mask = np.isin(self._metadata_df['celltype'], self._train_celltypes)
        val_celltype_mask = np.isin(self._metadata_df['celltype'], self._val_celltype)
        test_celltype_mask = np.isin(self._metadata_df['celltype'], self._test_celltype)
        
        split_array = -1*np.ones(self._metadata_df.shape[0]).astype(int)
        split_array[np.logical_and(train_chr_mask, train_celltype_mask)] = self._split_dict['train']
        split_array[np.logical_and(val_chr_mask, test_celltype_mask)] = self._split_dict['test']
        # Validate using test chr, either using a designated validation cell line ('val-ood') or a training cell line ('val-id')
        split_array[np.logical_and(val_chr_mask, val_celltype_mask)] = self._split_dict['val-ood']
        split_array[np.logical_and(val_chr_mask, train_celltype_mask)] = self._split_dict['val-id']
        if self._split_scheme=='standard':
            self._metadata_df['split'] = split_array
            self._split_array = split_array
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['celltype'])
        self._metric = Auprc()
        
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        Computes this from: 
        (1) sequence features in self._seq_bp
        (2) DNase features in self._dnase_allcelltypes
        (3) Metadata for the index (location along the genome with 1kb window width)
        """
        this_metadata = self._metadata_df.iloc[idx, :]
        flank_size = 400
        interval_start = this_metadata['start'] - flank_size
        interval_end = this_metadata['stop'] + flank_size
        dnase_this = _dnase_allcelltypes[this_metadata['celltype']][this_metadata['chr']][interval_start:interval_end]
        seq_this = _seq_bp[this_metadata['chr']][interval_start:interval_end]
        return np.column_stack([seq_this, dnase_this])

    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
