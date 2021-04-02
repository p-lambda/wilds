## ENCODE-TFBS-wilds feature generation and preprocessing

#### Requirements
- pyBigWig

#### Instructions

1. Download the human genome sequence (hg19 assembly) in FASTA format from http://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz and extract it into `SEQUENCE_PATH`.

2. Run `python prep_sequence.py --seq_path SEQUENCE_PATH --output_dir OUTPUT_DIR` to write the fasta file found in `SEQUENCE_PATH` to a numpy array archive in `OUTPUT_DIR`.

3. Download the DNase accessibility data. This consists of whole-genome DNase files in bigwig format from https://guanfiles.dcmb.med.umich.edu/Leopard/dnase_bigwig/.

4. Download the labels from the challenge into a label directory created for this purpose:
  - The training labels from https://www.synapse.org/#!Synapse:syn7413983 for the relevant transcription factor ( https://www.synapse.org/#!Synapse:syn7415202 for the TF MAX).
  - The validation labels from https://www.synapse.org/#!Synapse:syn8441154 for the relevant transcription factor ( https://www.synapse.org/#!Synapse:syn8442103 for the TF MAX). 
  - (Optional) The validation labels for the challenge's evaluation cell type (liver) from https://www.synapse.org/#!Synapse:syn8442975 for the relevant transcription factor ( https://www.synapse.org/#!Synapse:syn8443021 for the TF MAX).

5. Run `prep_metadata_labels.py`.
