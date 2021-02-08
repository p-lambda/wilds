import argparse, time
import numpy, pandas

from tqdm import tqdm


def one_hot_encode(sequence, ignore='N', alphabet=None, dtype='int8', 
	verbose=False, **kwargs):
	"""Converts a string or list of characters into a one-hot encoding.
	This function will take in either a string or a list and convert it into a
	one-hot encoding. If the input is a string, each character is assumed to be
	a different symbol, e.g. 'ACGT' is assumed to be a sequence of four 
	characters. If the input is a list, the elements can be any size.
	Although this function will be used here primarily to convert nucleotide
	sequences into one-hot encoding with an alphabet of size 4, in principle
	this function can be used for any types of sequences.
	Parameters
	----------
	sequence : str or list
		The sequence to convert to a one-hot encoding.
	ignore : str, optional
		A character to indicate setting nothing to 1 for that row, keeping the
		encoding entirely 0's for that row. In the context of genomics, this is
		the N character. Default is 'N'.
	alphabet : set or tuple or list, optional
		A pre-defined alphabet. If None is passed in, the alphabet will be
		determined from the sequence, but this may be time consuming for
		large sequences. Default is None.
	dtype : str or numpy.dtype, optional
		The data type of the returned encoding. Default is int8.
	verbose : bool or str, optional
		Whether to display a progress bar. If a string is passed in, use as the
		name of the progressbar. Default is False.
	kwargs : arguments
		Arguments to be passed into tqdm. Default is None.
	Returns
	-------
	ohe : numpy.ndarray
		A binary matrix of shape (alphabet_size, sequence_length) where
		alphabet_size is the number of unique elements in the sequence and
		sequence_length is the length of the input sequence.
	"""

	name = None if verbose in (True, False) else verbose
	d = verbose is False

	if isinstance(sequence, str):
		sequence = list(sequence)

	alphabet = alphabet or numpy.unique(sequence)
	alphabet = [char for char in alphabet if char != ignore]
	alphabet_lookup = {char: i for i, char in enumerate(alphabet)}

	ohe = numpy.zeros((len(sequence), len(alphabet)), dtype=dtype)
	for i, char in tqdm(enumerate(sequence), disable=d, desc=name, **kwargs):
		if char != ignore:
			idx = alphabet_lookup[char]
			ohe[i, idx] = 1

	return ohe


def read_fasta(filename, include_chroms=None, exclude_chroms=None, 
	ignore='N', alphabet=['A', 'C', 'G', 'T', 'N'], verbose=True):
	"""Read in a FASTA file and output a dictionary of sequences.
	This function will take in the path to a FASTA-formatted file and output
	a string containing the sequence for each chromosome. Optionally,
	the user can specify a set of chromosomes to include or exclude from
	the returned dictionary.
	Parameters
	----------
	filename : str
		The path to the FASTA-formatted file to open.
	include_chroms : set or tuple or list, optional
		The exact names of chromosomes in the FASTA file to include, excluding
		all others. If None, include all chromosomes (except those specified by
		exclude_chroms). Default is None.
	exclude_chroms : set or tuple or list, optional
		The exact names of chromosomes in the FASTA file to exclude, including
		all others. If None, include all chromosomes (or the set specified by
		include_chroms). Default is None.
	ignore : str, optional
		A character to indicate setting nothing to 1 for that row, keeping the
		encoding entirely 0's for that row. In the context of genomics, this is
		the N character. Default is 'N'.
	alphabet : set or tuple or list, optional
		A pre-defined alphabet. If None is passed in, the alphabet will be
		determined from the sequence, but this may be time consuming for
		large sequences. Must include the ignore character. Default is
		['A', 'C', 'G', 'T', 'N'].
	verbose : bool or str, optional
		Whether to display a progress bar. If a string is passed in, use as the
		name of the progressbar. Default is False.
	Returns
	-------
	chroms : dict
		A dictionary of strings where the keys are the names of the
		chromosomes (exact strings from the header lines in the FASTA file)
		and the values are the strings encoded there.
	"""

	sequences = {}
	name, sequence = None, None
	skip_chrom = False

	with open(filename, "r") as infile:
		for line in tqdm(infile, disable=not verbose):
			if line.startswith(">"):
				if name is not None and skip_chrom is False:
					sequences[name] = ''.join(sequence)

				sequence = []
				name = line[1:].strip("\n")
				if include_chroms is not None and name not in include_chroms:
					skip_chrom = True
				elif exclude_chroms is not None and name in exclude_chroms:
					skip_chrom = True
				else:
					skip_chrom = False

			else:
				if skip_chrom == False:
					sequence.append(line.rstrip("\n").upper())

	return sequences


def generate_sequence_archive(seq_path='sequence/hg19.genome.fa', output_dir):
    fasta_contents = read_fasta()
    kw_dict = {}
    itime = time.time()
    for chrom in chr_IDs:
        seqstr = fasta_contents[chrom]
        kw_dict[chrom] = one_hot_encode(seqstr, alphabet=['A', 'C', 'G', 'T', 'N'])
        print(chrom, time.time() - itime)

    # Save as npz archive; can take several (>20) minutes
    print("Saving npz archive...")
    np.savez_compressed('{}/sequence'.format(output_root), **kw_dict)
    print(time.time() - itime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    generate_sequence_archive(
        seq_path=args.seq_path,
        output_dir=args.output_dir)