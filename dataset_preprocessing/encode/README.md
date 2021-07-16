## ENCODE feature generation and preprocessing

#### Requirements
- pyBigWig

#### Instructions to create Codalab bundle

Here are instructions to reproduce the Codalab bundle, in a directory path `BUNDLE_ROOT_DIRECTORY`.

1. Download the human genome sequence (hg19 assembly) in FASTA format from http://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz and extract it into `SEQUENCE_PATH`.

2. Run `python prep_sequence.py --seq_path SEQUENCE_PATH --output_dir OUTPUT_DIR` to write the fasta file found in `SEQUENCE_PATH` to a numpy array archive in `OUTPUT_PATH`. (The dataset loader assumes `OUTPUT_PATH` to be `<bundle root directory>/sequence.npz`.)

3. Download the DNase accessibility data. This consists of whole-genome DNase files in bigwig format from https://guanfiles.dcmb.med.umich.edu/Leopard/dnase_bigwig/. Save these to filenames `<bundle root directory>/DNASE.<celltype>.fc.signal.bigwig` in the code.

4. Run `python prep_accessibility.py`. This writes samples of each bigwig file to `<bundle root directory>/qn.<celltype>.npy`. These are used at runtime when the dataset loader is initialized, to perform quantile normalization on the DNase accessibility signals.

5. Download the labels from the challenge into a label directory `<bundle root directory>/labels/` created for this purpose:
  - The training chromosome labels for the challenge's training cell types from https://www.synapse.org/#!Synapse:syn7413983 for the relevant transcription factor ( https://www.synapse.org/#!Synapse:syn7415202 for the TF MAX, downloaded as MAX.train.labels.tsv.gz ).
  - The training chromosome labels for the challenge's evaluation cell type (liver) from https://www.synapse.org/#!Synapse:syn8077511 for the relevant transcription factor ( https://www.synapse.org/#!Synapse:syn8077648 for the TF MAX, downloaded as MAX.train_wc.labels.tsv.gz ).
  - The validation chromosome labels for the challenge's training cell types from https://www.synapse.org/#!Synapse:syn8441154 for the relevant transcription factor ( https://www.synapse.org/#!Synapse:syn8442103 for the TF MAX, downloaded as MAX.val.labels.tsv.gz ).
  - The validation chromosome labels for the challenge's evaluation cell type (liver) from https://www.synapse.org/#!Synapse:syn8442975 for the relevant transcription factor ( https://www.synapse.org/#!Synapse:syn8443021 for the TF MAX, downloaded as MAX.test.labels.tsv.gz ).

6. Run `python prep_metadata_labels.py`.

