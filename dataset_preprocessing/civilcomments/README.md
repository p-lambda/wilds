## CivilComments-wilds processing

#### Instructions

1. Download `all_data.csv` from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data.

2. Run `python process_labeled.py --root ROOT`, where `ROOT` is where you downloaded `ROOT`. This will create `all_data_with_identities.csv` in the same folder, which is the labeled data that we use in WILDS.

3. After the above step, run `python process_unlabeled.py --root ROOT`, where `ROOT` is where you downloaded `ROOT`. This will create `unlabeled_data_with_identities.csv` in the same folder, which is the unlabeled data that we optionally use in WILDS.
