import os, time
import torch
import pandas as pd
import numpy as np
import pyBigWig
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.utils import subsample_idxs
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import MultiTaskAveragePrecision

# Human chromosomes in hg19
chrom_sizes = {'chr1': 249250621, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr2': 243199373, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chrX': 155270560}

# quantile normalization via numpy inter/extra-polation
def anchor(input_data, sample, ref): # input 1d array
    sample.sort()
    ref.sort()
    # 0. create the mapping function
    index = np.array(np.where(np.diff(sample) != 0)) + 1
    index = index.flatten()
    x = np.concatenate((np.zeros(1), sample[index])) # domain
    y = np.zeros(len(x)) # codomain
    for i in np.arange(0,len(index)-1, 1):
        start = index[i]
        end = index[i+1]
        y[i+1] = np.mean(ref[start:end])
    i += 1
    start = index[i]
    end = len(ref)
    y[i+1] = np.mean(ref[start:end])
    # 1. interpolate
    output = np.interp(input_data, x, y)
    # 2. extrapolate
    degree = 1 # degree of the fitting polynomial
    num = 10 # number of positions for extrapolate
    f1 = np.poly1d(np.polyfit(sample[-num:],ref[-num:],degree))
    output[input_data > sample[-1]] = f1(input_data[input_data > sample[-1]])
    return output


def wrap_anchor(
    signal,
    sample,
    ref
):
    ## 1.format as bigwig first
    x = signal
    z = np.concatenate(([0],x,[0])) # pad two zeroes
    # find boundary
    starts = np.where(np.diff(z) != 0)[0]
    ends = starts[1:]
    starts = starts[:-1]
    vals = x[starts]
    if starts[0] != 0:
        ends = np.concatenate(([starts[0]],ends))
        starts = np.concatenate(([0],starts))
        vals = np.concatenate(([0],vals))
    if ends[-1] != len(signal):
        starts = np.concatenate((starts,[ends[-1]]))
        ends = np.concatenate((ends,[len(signal)]))
        vals = np.concatenate((vals,[0]))

    ## 2.then quantile normalization
    vals_anchored = anchor(vals, sample, ref)
    return vals_anchored, starts, ends


def dnase_normalize(
    input_bw_celltype,
    ref_celltypes,
    out_fname,
    data_pfx
):
    if not data_pfx.endswith('/'):
        data_pfx = data_pfx + '/'
    itime = time.time()
    sample = np.load(data_pfx + "qn.{}.npy".format(input_bw_celltype))
    ref = np.zeros(len(sample))
    for ct in ref_celltypes:
        ref += (1.0/len(ref_celltypes))*np.load(data_pfx + "qn.{}.npy".format(ct))

    chromsizes_list = [(k, v) for k, v in chrom_sizes.items()]
    bw_output = pyBigWig.open(out_fname, 'w')
    bw_output.addHeader(chromsizes_list)

    for the_chr in chrom_sizes:
        signal = np.zeros(chrom_sizes[the_chr])
        bw = pyBigWig.open(data_pfx + 'DNASE.{}.fc.signal.bigwig'.format(input_bw_celltype))
        signal += np.nan_to_num(np.array(bw.values(the_chr, 0, chrom_sizes[the_chr])))
        bw.close()
        vals_anchored, starts, ends = wrap_anchor(signal, sample, ref)
        # write normalized dnase file.
        chroms = np.array([the_chr] * len(vals_anchored))
        bw_output.addEntries(chroms, starts, ends=ends, values=vals_anchored)
        print(input_bw_celltype, the_chr, time.time() - itime)

    bw_output.close()


class EncodeDataset(WILDSDataset):
    """
    ENCODE dataset of transcription factor binding sites.
    This is a subset of the dataset from the ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge.

    Note: The first time this dataset is used, it will run some one-off preprocessing scripts that will take some additional time.
    These scripts might cause a race condition if multiple jobs are started in parallel,
    so we recommend running a single job the first time you use this dataset.

    Supported `split_scheme`:
        - 'official'
        - 'test-to-test'

    Input (x):
        12800-base-pair regions of sequence with a quantified chromatin accessibility readout.

    Label (y):
        y is a 128-bit vector, with each element y_i indicating the binding status of a 200bp window. It is 1 if this 200bp region is bound by the transcription factor, and 0 otherwise, for i = 0,1,...,127.

        Concretely, suppose the input window x starts at coordinate sc, extending until coordinate (sc+12800). Then y_i is the label of the window starting at coordinate (sc+3200)+(50*i).

    Metadata:
        Each sequence is annotated with the celltype of origin (a string) and the chromosome of origin (a string).

    Website:
        https://www.synapse.org/#!Synapse:syn6131484 . This is the website for the challenge; the data can be downloaded from here as per the instructions in dataset_preprocessing/encode/README.md.
    """

    _dataset_name = 'encode'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x9c282b6e9082440f9dcd61bb605c1eab/contents/blob/',
            'compressed_size': 7_692_640_256}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        itime = time.time()
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._y_size = 128

        # Construct splits
        train_chroms = ['chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr22', 'chrX']
        val_chroms = ['chr2', 'chr9', 'chr11']
        test_chroms = ['chr1', 'chr8', 'chr21']
        official_train_cts = {
            'MAX': ['H1-hESC', 'HCT116', 'HeLa-S3', 'K562', 'A549', 'GM12878'],
            'JUND': ['HCT116', 'HeLa-S3', 'K562', 'MCF-7']
        }
        official_val_cts = {
            'MAX': ['HepG2'], 'JUND': ['HepG2']
        }
        official_test_cts = {
            'MAX': ['liver'], 'JUND': ['liver']
        }

        # Set the TF in split_scheme by prefacing it with 'tf.<TF name>.'
        self._transcription_factor = 'MAX'
        if 'tf.' in split_scheme:
            tkns = split_scheme.split('.')
            self._transcription_factor = tkns[1]
            split_scheme = '.'.join(tkns[2:])
        self._split_scheme = split_scheme

        train_celltypes = official_train_cts[self._transcription_factor]
        val_celltype = official_val_cts[self._transcription_factor]
        test_celltype = official_test_cts[self._transcription_factor]

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
                'id_test': {
                    'chroms': test_chroms,
                    'celltypes': train_celltypes
                }
            }
            self._split_dict = {
                'train': 0,
                'val': 1,
                'test': 2,
                'id_val': 3,
                'id_test': 4
            }
            self._split_names = {
                'train': 'Train',
                'val': 'Validation (OOD)',
                'test': 'Test',
                'id_val': 'Validation (ID)',
                'id_test': 'Test (ID)',
            }
        elif self._split_scheme == 'test-to-test':
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
        elif 'id-' in self._split_scheme:
            test_celltype = [ self._split_scheme.split('id-')[1] ]
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

        # Add new split scheme specifying custom test and val celltypes in the format val.<val celltype>.test.<test celltype>, e.g. self._split_scheme == 'official' is equivalent to self._split_scheme == 'val.HepG2.test.liver'
        elif '.' in self._split_scheme:
            all_celltypes = train_celltypes + val_celltype + test_celltype
            in_val_ct = self._split_scheme.split('.')[1]
            in_test_ct = self._split_scheme.split('.')[3]
            train_celltypes = [ct for ct in all_celltypes if ((ct != in_val_ct) and (ct != in_test_ct))]
            val_celltype = [in_val_ct]
            test_celltype = [in_test_ct]
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
                'id_test': {
                    'chroms': test_chroms,
                    'celltypes': train_celltypes
                }
            }
            self._split_dict = {
                'train': 0,
                'val': 1,
                'test': 2,
                'id_val': 3,
                'id_test': 4
            }
            self._split_names = {
                'train': 'Train',
                'val': 'Validation (OOD)',
                'test': 'Test',
                'id_val': 'Validation (ID)',
                'id_test': 'Test (ID)',
            }
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # Read in metadata and labels
        self._metadata_df = pd.read_csv(
            self._data_dir + '/labels/{}/metadata_df.bed'.format(self._transcription_factor),
            sep='\t', header=None,
            index_col=None, names=['chr', 'start', 'stop', 'celltype']
        )
        self._y_array = torch.tensor(np.load(
            self._data_dir + '/labels/{}/metadata_y.npy'.format(self._transcription_factor)))

        # ~10% of the dataset has ambiguous labels, i.e., we can't tell if there is a binding event or not. This typically happens at the flanking regions of peaks. For our purposes, we will ignore these ambiguous labels during training and eval.
        self.y_array[self.y_array == 0.5] = float('nan')

        self._split_array = -1 * np.ones(self._metadata_df.shape[0]).astype(int)
        for split, d in splits.items():
            chrom_mask = np.isin(self._metadata_df['chr'], d['chroms'])
            celltype_mask = np.isin(self._metadata_df['celltype'], d['celltypes'])
            self._split_array[chrom_mask & celltype_mask] = self._split_dict[split]

        keep_mask = (self._split_array != -1)

        # Remove all-zero sequences from training.
        train_mask = (self._split_array == self._split_dict['train'])
        allzeroes_mask = (self._y_array.sum(axis=1) == 0).numpy()
        keep_mask = keep_mask & ~(train_mask & allzeroes_mask)

        # Subsample the testing and validation indices, to speed up evaluation.
        # For the OOD splits (val and test), we subsample by a factor of 3
        # For the id_val and id_test splits, we subsample by a factor of 3*(# of training celltypes)
        for subsample_seed, (split, subsample_factor) in enumerate([
            ('val', 3),
            ('test', 3),
            ('id_val', 3*len(splits['train']['celltypes'])),
            ('id_test', 3*len(splits['train']['celltypes']))]):
            if split not in self._split_dict: continue
            split_mask = (self._split_array == self._split_dict[split])
            split_idxs = np.arange(len(self._split_array))[split_mask]
            idxs_to_remove = subsample_idxs(
                split_idxs,
                num=len(split_idxs) // subsample_factor,
                seed=subsample_seed,
                take_rest=True)
            keep_mask[idxs_to_remove] = False

        self._metadata_df = self._metadata_df[keep_mask]
        self._split_array = self._split_array[keep_mask]
        self._y_array = self._y_array[keep_mask]

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

        # Set up file handles for DNase features, writing normalized DNase tracks along the way if they aren't already written.
        self._dnase_allcelltypes = {}
        for ct in self._all_celltypes:
            orig_dnase_bw_path = os.path.join(self._data_dir, 'DNASE.{}.fc.signal.bigwig'.format(ct))
            dnase_bw_path = os.path.join(self._data_dir, 'DNase.{}.{}.{}.bigwig'.format(self._transcription_factor, ct, self._split_scheme))
            if not os.path.exists(dnase_bw_path):
                ref_celltypes = splits['train']['celltypes']
                dnase_normalize(ct, ref_celltypes, out_fname=dnase_bw_path, data_pfx=self._data_dir)
            self._dnase_allcelltypes[ct] = pyBigWig.open(dnase_bw_path)

        # Load subsampled DNase arrays for normalization purposes
        self._dnase_qnorm_arrays = {}
        for ct in self._all_celltypes:
            qnorm_arr_path = os.path.join(self._data_dir, 'qn.{}.npy'.format(ct))
            self._dnase_qnorm_arrays[ct] = np.load(qnorm_arr_path)
        self._norm_ref_distr = np.zeros(len(self._dnase_qnorm_arrays[ct]))
        test_cts = splits['test']['celltypes']
        num_to_avg = len(self._all_celltypes) - len(test_cts)
        for ct in self._all_celltypes:
            if ct not in test_cts:
                self._norm_ref_distr += (1.0/num_to_avg)*self._dnase_qnorm_arrays[ct]

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
        dnase_this = np.nan_to_num(dnase_bw.values(chrom, interval_start, interval_end, numpy=True))
        return torch.tensor(np.column_stack(
            [seq_this,
             dnase_this]
        ).T)


    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
