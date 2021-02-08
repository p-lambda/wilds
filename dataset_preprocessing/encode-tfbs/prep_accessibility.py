import numpy, pandas
import pyBigWig

from tqdm import tqdm


def generate_accessibility_archives(input_dir, output_dir):
    dnases = {}
    celltypes = ['A549', 'GM12878', 'H1-hESC', 'HCT116', 'HeLa-S3', 'HepG2', 'K562']

    for ctype in celltypes:#glob.glob('dnase_bigwigs/*'):
        itime = time.time()
        # ctype = pth.split('/')[1].split('.')[1]
        bw = pyBigWig.open("{}/DNASE.{}.fc.signal.bigwig".format(input_dir, ctype))
        chromsizes = bw.chroms()
        print(ctype, time.time() - itime)
        dn_dict = {}
        for chrom in chromsizes: #chr_IDs:
            x = bw.values(chrom, 0, chromsizes[chrom], numpy=True)
            dn_dict[chrom] = np.nan_to_num(x).astype(np.float16)   # half-precision makes things significantly smaller (less time to load)
            print(chrom, time.time() - itime)
        dnases[ctype] = dn_dict

    for ctype in dnases:
        itime = time.time()
        dn_dict = dnases[ctype]

        # Save as npz archive
        np.savez_compressed('{}/{}_dnase'.format(output_dir, ctype), **dn_dict)
        print("Saving npz archive for celltype {}. Time: {}".format(ctype, time.time() - itime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    generate_accessibility_archives(
        input_dir=args.input_dir,
        output_dir=args.output_dir)