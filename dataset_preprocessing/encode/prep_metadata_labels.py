import os, csv
import scipy, numpy as np, pandas as pd, time
from scipy import sparse
import pyBigWig

# Human chromosome names
chr_IDs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
           'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
           'chr20', 'chr21', 'chr22', 'chrX']
chrom_sizes = {'chr1': 249250621, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr2': 243199373, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chrX': 155270560}

_data_dir = '../../examples/data/encode_v1.0/'


def write_label_bigwigs(
    celltypes,
    train_suffix='train.labels.tsv.gz',
    val_suffix='val.labels.tsv.gz',
    tf_name='MAX'
):
    itime = time.time()

    # Read in metadata dataframe from training+validation data
    train_regions_labeled = pd.read_csv(os.path.join(_data_dir, 'labels/{}.{}'.format(tf_name, train_suffix)), sep='\t')
    val_regions_labeled = pd.read_csv(os.path.join(_data_dir, 'labels/{}.{}'.format(tf_name, val_suffix)), sep='\t')
    training_df = train_regions_labeled
    val_df = val_regions_labeled
    all_df = pd.concat([training_df, val_df])

    # Get the y values, and remove negative labels by default.
    pd_list = []
    for ct in celltypes:
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
            tf_name, tf_name)
    _sorted_dir = _unsorted_dir.replace(
        '{}_posamb'.format(tf_name),
        '{}_posamb.sorted'.format(tf_name)
    )
    _metadata_df.to_csv(
        _unsorted_dir, sep='\t', header=False, index=False
    )
    print(time.time() - itime)

    # Sort bigwigs (as bed files) in order to convert to bigwig.
    os.system('sort -k1,1 -k2,2n {} > {}'.format(_unsorted_dir, _sorted_dir))
    mdf_posamb = pd.read_csv(
        _sorted_dir,
        sep='\t', header=None, index_col=None, names=['chr', 'start', 'stop', 'y', 'celltype']
    )

    # Write the binned labels to bigwig files,  genome-wide labels
    chromsizes_list = [(k, v) for k, v in chrom_sizes.items()]
    for ct in celltypes:
        ct_labels_bw_path = _data_dir + "labels/{}/{}_{}.bigwig".format(
            tf_name, tf_name, ct)
        df = mdf_posamb[mdf_posamb['celltype'] == ct]
        bw = pyBigWig.open(ct_labels_bw_path, "w")
        bw.addHeader(chromsizes_list)
        bw.addEntries(list(df['chr']), list(df['start']), ends=list(df['start']+50), values=list(df['y']))
        print(ct, time.time() - itime)
        bw.close()


def write_metadata_products(
    celltypes,
    bed_df_filename='metadata_df.bed',
    y_arr_filename='metadata_y.npy',
    stride=6400,
    tf_name='MAX',
    posamb_only=False
):
    itime = time.time()
    celltype_mdta = []
    celltype_labels = []
    if posamb_only:
        mdf_posamb = pd.read_csv(
            _data_dir + 'labels/{}/{}_posamb.sorted.bed'.format(tf_name, tf_name),
            sep='\t', header=None, index_col=None, names=['chr', 'start', 'stop', 'y', 'celltype']
        )
    # Retrieve only the windows containing positively/ambiguously labeled bins (if posamb_only==True), or all windows (if posamb_only==False).
    for ct in celltypes:
        ct_labels_bw_path = _data_dir + "labels/{}/{}_{}.bigwig".format(tf_name, tf_name, ct)
        df_construction = []
        mdta_labels = []
        bw = pyBigWig.open(ct_labels_bw_path)
        if posamb_only: # Retrieve only the windows containing positively/ambiguously labeled bins
            df = mdf_posamb[mdf_posamb['celltype'] == ct]
            df['window_start'] = stride*(df['start'] // stride)
            uniq_windows = np.unique(["{}:{}".format(x[0], x[1]) for x in zip(df['chr'], df['window_start'])])
            for u in uniq_windows:
                u_chr = u.split(':')[0]
                u_start = int(u.split(':')[1])
                u_end = u_start + stride
                x = np.nan_to_num(bw.values(u_chr, u_start, u_end, numpy=True))
                df_construction.append((u_chr, u_start, u_end))
                mdta_labels.append(x[np.arange(0, len(x), 50)])
        else:  # Retrieve all windows genome-wide
            for chrID in bw.chroms():
                chromsize = bw.chroms()[chrID]
                # Iterate over windows
                for startc in np.arange(int(stride/2), chromsize-(2*stride), stride):
                    u_end = startc + stride
                    if u_end > chromsize:
                        break
                    x = np.nan_to_num(bw.values(chrID, startc, u_end, numpy=True))
                    df_construction.append((chrID, startc, u_end))
                    mdta_labels.append(x[np.arange(0, len(x), 50)])
                print(ct, chrID, time.time() - itime)
        celltype_mdta_df = pd.DataFrame(df_construction, columns=['chr', 'start', 'stop'])
        celltype_mdta_df.insert(len(celltype_mdta_df.columns), 'celltype', ct)
        celltype_mdta.append(celltype_mdta_df)
        celltype_labels.append(np.stack(mdta_labels))
        print(ct, time.time() - itime)
        bw.close()
    print(time.time() - itime)

    all_metadata_df = pd.concat(celltype_mdta)
    all_metadata_df.to_csv(
        _data_dir + 'labels/{}/{}'.format(tf_name, bed_df_filename),
        sep='\t', header=False, index=False
    )
    np.save(_data_dir + 'labels/{}/{}'.format(tf_name, y_arr_filename), np.vstack(celltype_labels))


if __name__ == '__main__':
    tfs_to_celltypes = {
        'MAX': ['H1-hESC', 'HCT116', 'HeLa-S3', 'HepG2', 'K562', 'A549', 'GM12878', 'liver'], 
        'JUND': ['HCT116', 'HeLa-S3', 'HepG2', 'K562', 'MCF-7', 'liver']
    }
    for tf_name in tfs_to_celltypes:
        all_celltypes = tfs_to_celltypes[tf_name]
        write_label_bigwigs([x for x in all_celltypes if x != 'liver'], tf_name=tf_name)
        write_label_bigwigs(['liver'], train_suffix='train_wc.labels.tsv.gz', val_suffix='test.labels.tsv.gz', tf_name=tf_name)
        write_metadata_products(all_celltypes, tf_name=tf_name)
