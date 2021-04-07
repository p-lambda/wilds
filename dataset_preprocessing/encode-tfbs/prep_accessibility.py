# Adapted from https://github.com/GuanLab/Leopard/blob/master/data/quantile_normalize_bigwig.py

import argparse, time
import numpy as np
import pyBigWig

# Human chromosomes in hg19
chrom_sizes = {'chr1': 249250621, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr2': 243199373, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chrX': 155270560}


def qn_sample_to_array(
    input_celltypes, 
    input_chroms=None, 
    subsampling_ratio=1000, 
    data_pfx = '/users/abalsubr/wilds/examples/data/encode-tfbs_v1.0/'
):
    itime = time.time()
    if input_chroms is None:
        input_chroms = chrom_sizes.keys()
    qn_chrom_sizes = { k: chrom_sizes[k] for k in input_chroms }
    # chromosome-specific subsampling seeds
    chr_to_seed = {}
    i = 0
    for the_chr in qn_chrom_sizes:
        chr_to_seed[the_chr] = i
        i += 1

    # subsampling; multiple replicates are added
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
        print(the_chr, ct, time.time() - itime)

    if np.any(np.isnan(sample)):
        print('wtf! sample contains nan!')
    sample.sort()
    np.save(data_pfx + "qn.{}.npy".format('.'.join(input_celltypes)), sample)


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
#    f2=np.poly1d(np.polyfit(sample[:num],ref[:num],degree))
    output[input_data > sample[-1]] = f1(input_data[input_data > sample[-1]])
#    output[input_data<sample[0]]=f2(input_data[input_data<sample[0]])
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
    sample_celltype, 
    ref_celltypes, 
    out_fname = 'norm', 
    data_pfx = '/users/abalsubr/wilds/examples/data/encode-tfbs_v1.0/'
):
    itime = time.time()
    sample = np.load(data_pfx + "qn.{}.npy".format(sample_celltype))
    ref = np.zeros(len(sample))
    for ct in ref_celltypes:
        ref += (1.0/len(ref_celltypes))*np.load(data_pfx + "qn.{}.npy".format(ct))

    chromsizes_list = [(k, v) for k, v in chrom_sizes.items()]
    bw_output = pyBigWig.open(data_pfx + 'DNase.{}.{}.bigwig'.format(
        input_bw_celltype, out_fname), 'w')
    bw_output.addHeader(chromsizes_list)
    # bw_output.addHeader(list(zip(chr_all , num_bp)), maxZooms=0) # zip two turples
    
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

def generate_accessibility_archives(input_dir='dnase_bigwigs', output_dir='codalab_archive'):
    dnases = {}
    celltypes = ['A549', 'GM12878', 'H1-hESC', 'HCT116', 'HeLa-S3', 'HepG2', 'K562']

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
    train_chroms = ['chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr22', 'chrX']
    ch_train_celltypes = ['H1-hESC', 'HCT116', 'HeLa-S3', 'K562', 'A549', 'GM12878']
    ch_val_celltype = ['HepG2']
    ch_test_celltype = ['liver']
    ref_celltypes = ch_train_celltypes
    all_celltypes = ch_train_celltypes + ch_val_celltype + ch_test_celltype
    for ct in all_celltypes:
        qn_sample_to_array([ct], input_chroms=train_chroms)
    
    # Create normalized bigwigs for OOD validation split.
    for ct in all_celltypes:
        dnase_normalize(ct, ct, ref_celltypes)
    # Create normalized bigwig for ID validation split.
    for ct in ch_test_celltype:
        dnase_normalize(ct, ct, ch_test_celltype, out_fname = 'norm_id')
