import os, csv
import scipy, numpy as np, pandas as pd, time
from scipy import sparse
import pyBigWig

# Human chromosome names
chr_IDs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
chrom_sizes = {'chr1': 249250621, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr2': 243199373, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chrX': 155270560}

_data_dir = '../../examples/data/encode-tfbs_v1.0/'


def write_label_bigwigs():
    itime = time.time()
    transcription_factor = 'MAX'
    _train_chroms = ['chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr22', 'chrX']
    _val_chroms = ['chr2', 'chr9', 'chr11']
    _test_chroms = ['chr1', 'chr8', 'chr21']
    _all_chroms = _train_chroms + _val_chroms + _test_chroms
    _train_celltypes = ['H1-hESC', 'HCT116', 'HeLa-S3', 'HepG2', 'K562']
    _val_celltype = ['A549']
    _test_celltype = ['GM12878']
    _all_celltypes = _train_celltypes + _val_celltype + _test_celltype

    # Read in metadata dataframe from training+validation data
    train_regions_labeled = pd.read_csv(os.path.join(_data_dir, 'labels/{}.train.labels.tsv.gz'.format(_transcription_factor)), sep='\t')
    val_regions_labeled = pd.read_csv(os.path.join(_data_dir, 'labels/{}.val.labels.tsv.gz'.format(_transcription_factor)), sep='\t')
    training_df = train_regions_labeled# [np.isin(train_regions_labeled['chr'], _train_chroms)]
    val_df = val_regions_labeled# [np.isin(val_regions_labeled['chr'], _test_chroms)]
    all_df = pd.concat([training_df, val_df])

    print(time.time() - itime)

    # Get the y values, and remove labels by default.
    pd_list = []
    for ct in _all_celltypes:
        tc_chr = all_df[['chr', 'start', 'stop', ct]]
        tc_chr.columns = ['chr', 'start', 'stop', 'y']
        tc_chr = tc_chr[tc_chr['y'] != 'U']
        tc_chr['y'] = tc_chr['y'].replace({'U': 0, 'B': 1, 'A': 0.5}).values

        tc_chr.insert(len(tc_chr.columns), 'celltype', ct)
        pd_list.append(tc_chr)
        print(ct, time.time() - itime)
    _metadata_df = pd.concat(pd_list)

    print(time.time() - itime)
    _unsorted_dir = _data_dir + 'labels/{}/{}_posamb.bed'.format(
            transcription_factor, transcription_factor)
    _sorted_dir = _unsorted_dir.replace(
        '{}_posamb'.format(transcription_factor), 
        '{}_posamb.sorted'.format(transcription_factor)
    )
    _metadata_df.to_csv(
        _unsorted_dir, sep='\t', header=False, index=False
    )
    print(time.time() - itime)

    os.system('sort -k1,1 -k2,2n {} > {}'.format(_unsorted_dir, _sorted_dir))

    mdf_posamb = pd.read_csv(
        _sorted_dir, 
        sep='\t', header=None, index_col=None, names=['chr', 'start', 'stop', 'y', 'celltype']
    )
    
    # Write the binned labels to bigwig files - genome-wide labels
    chromsizes_list = [(k, v) for k, v in chrom_sizes.items()]
    for ct in _all_celltypes:
        ct_labels_bw_path = _data_dir + "labels/{}/{}_{}.bigwig".format(
            transcription_factor, transcription_factor, ct)
        df = mdf_posamb[mdf_posamb['celltype'] == ct]
        bw = pyBigWig.open(ct_labels_bw_path, "w")
        bw.addHeader(chromsizes_list)
        bw.addEntries(list(df['chr']), list(df['start']), ends=list(df['start']+50), values=list(df['y']))
        print(ct, time.time() - itime)
        bw.close()


def write_():
    stride = 6400
    itime = time.time()
    mdf_posamb = pd.read_csv(
        _sorted_dir, 
        sep='\t', header=None, index_col=None, names=['chr', 'start', 'stop', 'y', 'celltype']
    )
    celltype_mdta = []
    celltype_labels = []

    for ct in _all_celltypes:
        ct_labels_bw_path = _data_dir + "labels/MAX/MAX_{}.bigwig".format(ct)
        df = mdf_posamb[mdf_posamb['celltype'] == ct]
        df['window_start'] = stride*(df['start'] // stride)
        uniq_windows = np.unique(["{}:{}".format(x[0], x[1]) for x in zip(df['chr'], df['window_start'])])
        df_construction = []
        mdta_labels = []

        bw = pyBigWig.open(ct_labels_bw_path)
        num_reps = 0
        for u in uniq_windows:
            u_chr = u.split(':')[0]
            u_start = int(u.split(':')[1])
            u_end = u_start + stride
            x = np.nan_to_num(bw.values(u_chr, u_start, u_end, numpy=True))
            df_construction.append((u_chr, u_start, u_end))
            mdta_labels.append(x[np.arange(0, len(x), 50)])
            num_reps = num_reps + 1
        celltype_mdta_df = pd.DataFrame(df_construction, columns=['chr', 'start', 'stop'])
        celltype_mdta_df.insert(len(celltype_mdta_df.columns), 'celltype', ct)
        celltype_mdta.append(celltype_mdta_df)
        celltype_labels.append(np.stack(mdta_labels))
        print(ct, time.time() - itime)
        bw.close()
        # break
    print(time.time() - itime)
    # _metadata_df

    pd.concat(celltype_mdta).to_csv(
        _data_dir + 'labels/MAX/metadata_df.bed', 
        sep='\t', header=False, index=False
    )
    np.save(_data_dir + 'labels/MAX/metadata_y.npy', np.vstack(celltype_labels))
    print(time.time() - itime)


if __name__ == '__main__':
    write_label_bigwigs()
    