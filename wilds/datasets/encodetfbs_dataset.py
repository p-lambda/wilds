import os
import torch
import pandas as pd
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class EncodeTFBSDataset(WILDSDataset):
    """
    ENCODE-DREAM-wilds dataset of transcription factor binding sites. 
    This is a subset of the dataset from the ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge. 
    
    Input (x):
        1000-base-pair regions of sequence with a quantified chromatin accessibility readout.

    Label (y):
        y is binary. It is 1 if the central 200bp region is bound by the transcription factor MAX, and 0 otherwise.

    Metadata:
        Each sequence is annotated with the celltype of origin (a string) and the chromosome of origin (a string).
    
    Website:
        https://www.synapse.org/#!Synapse:syn6131484
    """

    def __init__(self, root_dir, download, split_scheme):
        self._dataset_name = 'encodeTFBS'
        self._download_url = 'https://worksheets.codalab.org/rest/bundles/0x8b3255e21e164cd98d3aeec09cd0bc26/contents/blob/'
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._y_size = 1
        self._n_classes = 2
        
        # self._tr_chrs = ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr22', 'chrX']
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
        (3) Metadata for the index (location along the genome with 200bp window width)
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
