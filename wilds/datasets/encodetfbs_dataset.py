import os, time
import torch
import pandas as pd
import numpy as np
import pyBigWig
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import MultiTaskAveragePrecision

class EncodeTFBSDataset(WILDSDataset):
    """
    ENCODE-DREAM-wilds dataset of transcription factor binding sites.
    This is a subset of the dataset from the ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge.

    Input (x):
        12800-base-pair regions of sequence with a quantified chromatin accessibility readout.

    Label (y):
        y is a 128-bit vector, with each element y_i indicating the binding status of a 200bp window. It is 1 if this 200bp region is bound by the transcription factor, and 0 otherwise. If the window x starts at coordinate sc, y_i is the label of the window starting at coordinate (sc+3200)+(50*i).

    Metadata:
        Each sequence is annotated with the celltype of origin (a string) and the chromosome of origin (a string).

    Website:
        https://www.synapse.org/#!Synapse:syn6131484
    """

    _dataset_name = 'encode-tfbs'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x7efd626149d648f699d9e686d7aa81a9/contents/blob/',
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        itime = time.time()
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._y_size = 128
        self._transcription_factor = 'MAX'

        # Read in metadata and labels
        self._metadata_df = pd.read_csv(
            self._data_dir + '/labels/{}/metadata_df.bed'.format(self._transcription_factor),
            sep='\t', header=None,
            index_col=None, names=['chr', 'start', 'stop', 'celltype']
        )
        self._y_array = torch.tensor(np.load(
            self._data_dir + '/labels/{}/metadata_y.npy'.format(self._transcription_factor)))

        # ~10% of the dataset has ambiguous labels
        # i.e., we can't tell if there is a binding event or not.
        # This typically happens at the flanking regions of peaks.
        # For our purposes, we will ignore these ambiguous labels during training and eval.
        self.y_array[self.y_array == 0.5] = float('nan')

        # Construct splits
        train_chroms = ['chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr22', 'chrX']
        val_chroms = ['chr2', 'chr9', 'chr11']
        test_chroms = ['chr1', 'chr8', 'chr21']
        train_celltypes = ['H1-hESC', 'HCT116', 'HeLa-S3', 'HepG2', 'K562']
        val_celltype = ['A549']
        test_celltype = ['GM12878']
        self._split_scheme = split_scheme
        if self._split_scheme == 'official':
            splits = {
                'train': {
                    'chroms': train_chroms,
                    'celltypes': train_celltypes
                },
                'id_val': {
                    'chroms': val_chroms,
                    'celltypes': train_celltypes
                },
                'val': {
                    'chroms': val_chroms,
                    'celltypes': val_celltype
                },
                'test': {
                    'chroms': test_chroms,
                    'celltypes': test_celltype
                },
            }
            self._split_dict = {
                'train': 0,
                'val': 1,
                'test': 2,
                'id_val': 3,
            }
            self._split_names = {
                'train': 'Train',
                'val': 'Validation (OOD)',
                'test': 'Test',
                'id_val': 'Validation (ID)',
            }
        elif self._split_scheme == 'in-dist':
            splits = {
                'train': {
                    'chroms': train_chroms,
                    'celltypes': test_celltype,
                },
                'val': {
                    'chroms': val_chroms,
                    'celltypes': test_celltype
                },
                'test': {
                    'chroms': test_chroms,
                    'celltypes': test_celltype
                },
            }
            self._split_dict = {
                'train': 0,
                'val': 1,
                'test': 2,
            }
            self._split_names = {
                'train': 'Train',
                'val': 'Validation (OOD)',
                'test': 'Test',
            }
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        self._split_array = -1 * np.ones(self._metadata_df.shape[0]).astype(int)
        for split, d in splits.items():
            chrom_mask = np.isin(self._metadata_df['chr'], d['chroms'])
            celltype_mask = np.isin(self._metadata_df['celltype'], d['celltypes'])
            self._split_array[chrom_mask & celltype_mask] = self._split_dict[split]

        indices_to_keep = (self._split_array != -1)
        # Remove all-zero sequences from training.
        remove_allnegative = True
        if remove_allnegative:
            train_msk = (self._split_array == self._split_dict['train'])
            allzeroes_msk = (self._y_array.sum(axis=1) == 0).numpy()
            indices_to_keep = indices_to_keep & ~(train_msk & allzeroes_msk)
        # Subsample the testing and validation indices
        val_msk = (self._split_array == self._split_dict['val'])
        test_msk = (self._split_array == self._split_dict['test'])
        idval_msk = (self._split_array == self._split_dict['id_val'])
        subsamp_factor_id = 15
        subsamp_factor_ood = 3
        
        keep_mask_ood = np.random.binomial(1, 1.0/subsamp_factor_ood, size=len(indices_to_keep)).astype(bool)
        indices_to_keep = indices_to_keep & ~(~keep_mask_ood & val_msk)
        indices_to_keep = indices_to_keep & ~(~keep_mask_ood & test_msk)
        
        keep_mask_id = np.random.binomial(1, 1.0/subsamp_factor_id, size=len(indices_to_keep)).astype(bool)
        indices_to_keep = indices_to_keep & ~(~keep_mask_id & idval_msk)
        
        self._metadata_df = self._metadata_df[indices_to_keep]
        self._split_array = self._split_array[indices_to_keep]
        self._y_array = self._y_array[indices_to_keep]

        self._all_chroms = sorted(list({chrom for _, d in splits.items() for chrom in d['chroms']}))
        self._all_celltypes = sorted(list({chrom for _, d in splits.items() for chrom in d['celltypes']}))

        # Load sequence into memory
        sequence_filename = os.path.join(self._data_dir, 'sequence.npz')
        seq_arr = np.load(sequence_filename)
        self._seq_bp = {}
        for chrom in self._all_chroms:
            self._seq_bp[chrom] = seq_arr[chrom]
            print(chrom, time.time() - itime)
        del seq_arr

        # Set up file handles for DNase features
        self._dnase_allcelltypes = {}
        for ct in self._all_celltypes:
            dnase_bw_path = os.path.join(self._data_dir, 'DNase/{}.bigwig'.format(ct))
            self._dnase_allcelltypes[ct] = pyBigWig.open(dnase_bw_path)

        # Set up metadata fields, map, array
        self._metadata_fields = ['chr', 'celltype']
        self._metadata_map = {}
        self._metadata_map['chr'] = self._all_chroms
        self._metadata_map['celltype'] = self._all_celltypes
        chr_ints = self._metadata_df['chr'].replace(dict( [(y, x) for x, y in enumerate(self._metadata_map['chr'])] )).values
        celltype_ints = self._metadata_df['celltype'].replace(dict( [(y, x) for x, y in enumerate(self._metadata_map['celltype'])] )).values
        self._metadata_array = torch.stack(
            (torch.LongTensor(chr_ints),
             torch.LongTensor(celltype_ints)
            ),
            dim=1)

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['celltype'])

        self._metric = MultiTaskAveragePrecision()

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx, window_size=12800):
        """
        Returns x for a given idx in metadata_array, which has been filtered to only take windows with the desired stride.
        Computes this from:
        (1) sequence features in self._seq_bp
        (2) DNase bigwig file handles in self._dnase_allcelltypes
        (3) Metadata for the index (location along the genome with 6400bp window width)
        (4) Window_size, the length of sequence returned (centered on the 6400bp region in (3))
        """
        this_metadata = self._metadata_df.iloc[idx, :]
        chrom = this_metadata['chr']
        interval_start = this_metadata['start'] - int(window_size/4)
        interval_end = interval_start + window_size
        seq_this = self._seq_bp[this_metadata['chr']][interval_start:interval_end]
        dnase_bw = self._dnase_allcelltypes[this_metadata['celltype']]
        try:
            dnase_this = dnase_bw.values(chrom, interval_start, interval_end, numpy=True)
        except RuntimeError:
            print("error", chrom, interval_start, interval_end)
        
        assert(np.isnan(seq_this).sum() == 0)
        assert(np.isnan(dnase_this).sum() == 0)
        return torch.tensor(np.column_stack(
            [seq_this,
             dnase_this]
        ).T)

    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
