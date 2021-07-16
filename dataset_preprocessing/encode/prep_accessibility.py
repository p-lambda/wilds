# Adapted from https://github.com/GuanLab/Leopard/blob/master/data/quantile_normalize_bigwig.py

import argparse, time
import numpy as np
import pyBigWig

# Human chromosomes in hg19, and their sizes in bp
chrom_sizes = {'chr1': 249250621, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr2': 243199373, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chrX': 155270560}


def qn_sample_to_array(
    input_celltypes,
    input_chroms=None,
    subsampling_ratio=1000,
    data_pfx = '/users/abalsubr/wilds/examples/data/encode_v1.0/'
):
    """
    Compute and write distribution of DNase bigwigs corresponding to input celltypes.
    """
    if input_chroms is None:
        input_chroms = chrom_sizes.keys()
    qn_chrom_sizes = { k: chrom_sizes[k] for k in input_chroms }
    # Initialize chromosome-specific seeds for subsampling
    chr_to_seed = {}
    i = 0
    for the_chr in qn_chrom_sizes:
        chr_to_seed[the_chr] = i
        i += 1

    # subsampling
    sample_len = np.ceil(np.array(list(qn_chrom_sizes.values()))/subsampling_ratio).astype(int)
    sample = np.zeros(sum(sample_len))
    start = 0
    j = 0
    for the_chr in qn_chrom_sizes:
        np.random.seed(chr_to_seed[the_chr])
        for ct in input_celltypes:
            path = data_pfx + 'DNASE.{}.fc.signal.bigwig'.format(ct)
            bw = pyBigWig.open(path)
            signal = np.nan_to_num(np.array(bw.values(the_chr, 0, qn_chrom_sizes[the_chr])))
            index = np.random.randint(0, len(signal), sample_len[j])
            sample[start:(start+sample_len[j])] += (1.0/len(input_celltypes))*signal[index]
        start += sample_len[j]
        j += 1
        print(the_chr, ct)
    sample.sort()
    np.save(data_pfx + "qn.{}.npy".format('.'.join(input_celltypes)), sample)


if __name__ == '__main__':
    train_chroms = ['chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr22', 'chrX']
    all_celltypes = ['H1-hESC', 'HCT116', 'HeLa-S3', 'K562', 'A549', 'GM12878', 'MCF-7', 'HepG2', 'liver']
    for ct in all_celltypes:
        qn_sample_to_array([ct], input_chroms=train_chroms)
