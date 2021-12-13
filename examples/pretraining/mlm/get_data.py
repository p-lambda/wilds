import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import csv

os.system('mkdir -p examples/pretraining/mlm/data')


######################## CivilComments ########################
CCU_metadata_df = pd.read_csv('data/civilcomments_unlabeled_v1.0/unlabeled_data_with_identities.csv', index_col=0)
CCU_text_array = list(CCU_metadata_df['comment_text']) #1_551_515

with open('examples/pretraining/mlm/data/civilcomments_train.json', 'w') as outf:
    for text in tqdm(CCU_text_array):
        print (json.dumps({'text': text}), file=outf)


CC_metadata_df = pd.read_csv('data/civilcomments_v1.0/all_data_with_identities.csv', index_col=0)
CC_text_array_val = list(CC_metadata_df[CC_metadata_df['split'] == 'val']['comment_text']) #45_180

with open('examples/pretraining/mlm/data/civilcomments_val.json', 'w') as outf:
    for text in tqdm(CC_text_array_val):
        print (json.dumps({'text': text}), file=outf)



######################## Amazon ########################
amazon_data_df: pd.DataFrame = pd.read_csv(
    'data/amazon_v2.1/reviews.csv',
    dtype={
        "reviewerID": str,
        "asin": str,
        "reviewTime": str,
        "unixReviewTime": int,
        "reviewText": str,
        "summary": str,
        "verified": bool,
        "category": str,
        "reviewYear": int,
    },
    keep_default_na=False,
    na_values=[],
    quoting=csv.QUOTE_NONNUMERIC,
) #10_116_947

amazon_split_df: pd.DataFrame = pd.read_csv('data/amazon_v2.1/splits/user.csv') #10_116_947
is_in_dataset: bool = (amazon_split_df["split"] != -1)

amazon_split_df = amazon_split_df[is_in_dataset] #4_002_170
amazon_data_df = amazon_data_df[is_in_dataset] #4_002_170

# "val_unlabeled": 11, "test_unlabeled": 12, "extra_unlabeled": 13, "val": 1
_text_array_11  = list(amazon_data_df[amazon_split_df['split']==11]['reviewText']) #266_066
_text_array_12  = list(amazon_data_df[amazon_split_df['split']==12]['reviewText']) #268_761
_text_array_13  = list(amazon_data_df[amazon_split_df['split']==13]['reviewText']) #2_927_841
_text_array_val = list(amazon_data_df[amazon_split_df['split']==1]['reviewText']) #100_050

with open('examples/pretraining/mlm/data/amazon_train_11.json', 'w') as outf:
    for text in tqdm(_text_array_11):
        print (json.dumps({'text': text}), file=outf)

with open('examples/pretraining/mlm/data/amazon_train_12.json', 'w') as outf:
    for text in tqdm(_text_array_12):
        print (json.dumps({'text': text}), file=outf)

with open('examples/pretraining/mlm/data/amazon_train_13.json', 'w') as outf:
    for text in tqdm(_text_array_13):
        print (json.dumps({'text': text}), file=outf)

with open('examples/pretraining/mlm/data/amazon_train_11_12_13.json', 'w') as outf:
    for text in tqdm(_text_array_11 + _text_array_12 + _text_array_13):
        print (json.dumps({'text': text}), file=outf)

with open('examples/pretraining/mlm/data/amazon_val.json', 'w') as outf:
    for text in tqdm(_text_array_val):
        print (json.dumps({'text': text}), file=outf)
