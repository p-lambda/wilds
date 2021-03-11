import os, time
import torch
import pandas as pd
import numpy as np
import pyBigWig
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy, MultiTaskAccuracy

all_chrom_names = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

class EncodeTFBSDataset(WILDSDataset):
    """
    ENCODE-DREAM-wilds dataset of transcription factor binding sites. 
    This is a subset of the dataset from the ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge. 
    
    Input (x):
        12800-base-pair regions of sequence with a quantified chromatin accessibility readout.

    Label (y):
        y is binary. It is 1 if the central 200bp region is bound by the transcription factor MAX, and 0 otherwise.

    Metadata:
        Each sequence is annotated with the celltype of origin (a string) and the chromosome of origin (a string).
    
    Website:
        https://www.synapse.org/#!Synapse:syn6131484
    """

    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        itime = time.time()
        self._dataset_name = 'encode-tfbs'
        self._version = '1.0'
        self._download_url = 'https://worksheets.codalab.org/rest/bundles/0x8b3255e21e164cd98d3aeec09cd0bc26/contents/blob/'
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._y_size = 128
        # self._n_classes = 2
        
        self._train_chroms = ['chr3']#, 'chr4', 'chr5', 'chr6', 'chr7', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr22', 'chrX']
        self._val_chroms = ['chr2']#, 'chr9', 'chr11']
        self._test_chroms = ['chr1']#, 'chr8', 'chr21']
        self._transcription_factor = 'MAX'
        self._train_celltypes = ['H1-hESC', 'HCT116', 'HeLa-S3', 'HepG2', 'K562']
        self._val_celltype = ['A549']
        self._test_celltype = ['GM12878']
        self._all_chroms = self._train_chroms + self._val_chroms + self._test_chroms
        self._all_celltypes = self._train_celltypes + self._val_celltype + self._test_celltype
        
        self._metadata_map = {}
        self._metadata_map['chr'] = self._all_chroms
        self._metadata_map['celltype'] = self._all_celltypes
        
        # Get the splits
        if split_scheme=='official':
            split_scheme = 'standard'
        
        self._split_scheme = split_scheme
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
        
        # Load sequence and DNase features
        sequence_filename = os.path.join(self._data_dir, 'sequence.npz')
        seq_arr = np.load(sequence_filename)
        self._seq_bp = {}
        for chrom in self._all_chroms: #seq_arr:
            self._seq_bp[chrom] = seq_arr[chrom]
            print(chrom, time.time() - itime)
        
        self._dnase_allcelltypes = {}
        ct = 'avg'
        dnase_avg_bw_path = os.path.join(self._data_dir, 'Leopard_dnase/{}.bigwig'.format(ct))
        self._dnase_allcelltypes[ct] = pyBigWig.open(dnase_avg_bw_path)
        for ct in self._all_celltypes:
            """
            dnase_filename = os.path.join(self._data_dir, '{}_dnase.npz'.format(ct))
            dnase_npz_contents = np.load(dnase_filename)
            self._dnase_allcelltypes[ct] = {}
            for chrom in self._all_chroms: #self._seq_bp:
                self._dnase_allcelltypes[ct][chrom] = dnase_npz_contents[chrom]
            """
            dnase_bw_path = os.path.join(self._data_dir, 'Leopard_dnase/{}.bigwig'.format(ct))
            self._dnase_allcelltypes[ct] = pyBigWig.open(dnase_bw_path)
        
        self._metadata_df = pd.read_csv(
            self._data_dir + '/labels/MAX/metadata_df.bed', sep='\t', header=None, 
            index_col=None, names=['chr', 'start', 'stop', 'celltype']
        )
        
        train_regions_mask = np.isin(self._metadata_df['chr'], self._train_chroms)
        val_regions_mask = np.isin(self._metadata_df['chr'], self._val_chroms)
        test_regions_mask = np.isin(self._metadata_df['chr'], self._test_chroms)
        train_celltype_mask = np.isin(self._metadata_df['celltype'], self._train_celltypes)
        val_celltype_mask = np.isin(self._metadata_df['celltype'], self._val_celltype)
        test_celltype_mask = np.isin(self._metadata_df['celltype'], self._test_celltype)
        
        split_array = -1*np.ones(self._metadata_df.shape[0]).astype(int)
        split_array[np.logical_and(train_regions_mask, train_celltype_mask)] = self._split_dict['train']
        split_array[np.logical_and(test_regions_mask, test_celltype_mask)] = self._split_dict['test']
        # Validate using validation chr, either using a designated validation cell line ('val') or a training cell line ('id_val')
        split_array[np.logical_and(val_regions_mask, val_celltype_mask)] = self._split_dict['val']
        split_array[np.logical_and(val_regions_mask, train_celltype_mask)] = self._split_dict['id_val']
        
        if self._split_scheme=='standard':
            self._metadata_df.insert(len(self._metadata_df.columns), 'split', split_array)
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        
        metadata_mask = (self._metadata_df['split'] != -1)
        self._metadata_df = self._metadata_df[self._metadata_df['split'] != -1]
        
        chr_ints = self._metadata_df['chr'].replace(dict( [(y, x) for x, y in enumerate(self._metadata_map['chr'])] )).values
        celltype_ints = self._metadata_df['celltype'].replace(dict( [(y, x) for x, y in enumerate(self._metadata_map['celltype'])] )).values
        self._split_array = self._metadata_df['split'].values
        self._y_array = torch.Tensor(np.load(self._data_dir + '/labels/MAX/metadata_y.npy'))
        self._y_array = self._y_array[metadata_mask]
        
        self._metadata_array = torch.stack(
            (torch.LongTensor(chr_ints), 
             torch.LongTensor(celltype_ints)
            ),
            dim=1)
        self._metadata_fields = ['chr', 'celltype']
        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['celltype'])
        
        self._metric = MultiTaskAccuracy()
        
        super().__init__(root_dir, download, split_scheme)
    
    """
    def get_random_label_vec(metadata_df, output_size=128):
        # Sample a positively labeled region at random
        pos_mdf = metadata_df[metadata_df['y'] == 1] #.iloc[ metadata_df['chr'] == s['chr'], : ]
        pos_seed_region = pos_mdf.iloc[np.random.randint(pos_mdf.shape[0])]

        # Extract regions from this chromosome in this celltype, to get a window of labels from
        chr_msk = np.array(metadata_df['chr']) == pos_seed_region['chr']
        ct_msk = np.array(metadata_df['celltype']) == pos_seed_region['celltype']
        mdf = metadata_df[chr_msk & ct_msk]

        # Get labels
        start_ndx = np.where(mdf['start'] == pos_seed_region['start'])[0][0]
        y_label_vec = mdf.iloc[start_ndx:start_ndx+output_size, :]['y']
    """
    
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
        interval_end = interval_start + window_size  #this_metadata['stop']
        seq_this = self._seq_bp[this_metadata['chr']][interval_start:interval_end]
        dnase_bw = self._dnase_allcelltypes[this_metadata['celltype']]
        dnase_this = dnase_bw.values(chrom, interval_start, interval_end, numpy=True)
        # print("{}:{}-{}".format(chrom, interval_start, interval_end))
        dnase_avg = self._dnase_allcelltypes['avg'].values(chrom, interval_start, interval_end, numpy=True)
        return torch.tensor(np.column_stack(
            [np.nan_to_num(seq_this), 
             np.nan_to_num(dnase_this), np.nan_to_num(dnase_avg)]
        ).T)

    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
