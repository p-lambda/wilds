import argparse, time
import numpy as np
import pyBigWig

# Human hg19 chromosome names/lengths
chrom_sizes = {'chr1': 249250621, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr2': 243199373, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chrX': 155270560}

celltypes = ['A549', 'GM12878', 'H1-hESC', 'HCT116', 'HeLa-S3', 'HepG2', 'K562']

chr_IDs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

def write_label_bigwig(
    metadata_df, output_dir='codalab_archive'
):
    dnases = {}
    for ctype in celltypes:
        itime = time.time()
        bw = pyBigWig.open("{}/DNASE.{}.fc.signal.bigwig".format(input_dir, ctype))
        chromsizes = bw.chroms()
        dn_dict = {}
        for chrom in chromsizes: #chr_IDs:
            x = bw.values(chrom, 0, chromsizes[chrom], numpy=True)
            # half-precision makes things significantly smaller (less time to load)
            dn_dict[chrom] = np.nan_to_num(x).astype(np.float16)
            print("{}, {}. Time: {}".format(ctype, chrom, time.time() - itime))
        dnases[ctype] = dn_dict

    for ctype in dnases:
        itime = time.time()
        dn_dict = dnases[ctype]

        # Save as npz archive
        np.savez_compressed('{}/{}_dnase'.format(output_dir, ctype), **dn_dict)
        print("Saving npz archive for celltype {}. Time: {}".format(ctype, time.time() - itime))


if __name__ == '__main__':
    itime = time.time()
    _data_dir = '../../examples/data/encode-tfbs_v1.0/'
    _transcription_factor = 'MAX'
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
    _unsorted_dir = _data_dir + 'labels/MAX/MAX_posamb.bed'
    _sorted_dir = _unsorted_dir.replace('MAX_posamb', 'MAX_posamb.sorted')
    _metadata_df.to_csv(
        _unsorted_dir, sep='\t', header=False, index=False
    )
    print(time.time() - itime)

    os.system('sort -k1,1 -k2,2n {} > {}'.format(_unsorted_dir, _sorted_dir))
    
    mdf_posamb = pd.read_csv(
        _sorted_dir, 
        sep='\t', header=None, index_col=None, names=['chr', 'start', 'stop', 'y', 'celltype']
    )
    chromsizes_list = [(k, v) for k, v in chrom_sizes.items()]
    for ct in _all_celltypes:
        ct_labels_bw_path = _data_dir + "labels/MAX/MAX_{}.bigwig".format(ct)
        df = mdf_posamb[mdf_posamb['celltype'] == ct]
        bw = pyBigWig.open(ct_labels_bw_path, "w")
        bw.addHeader(chromsizes_list)
        bw.addEntries(list(df['chr']), list(df['start']), ends=list(df['start']+50), values=list(df['y']))
        print(ct, time.time() - itime)
        bw.close()
