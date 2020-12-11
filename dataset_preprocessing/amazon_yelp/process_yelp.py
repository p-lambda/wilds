import os, sys, torch, json, csv, argparse
import numpy as np
import pandas as pd 
from transformers import BertTokenizerFast
from utils import *

#############
### PATHS ###
#############

def data_dir(root_dir):
    return os.path.join(root_dir, 'yelp', 'data')

def token_length_path(data_dir):
    return os.path.join(preprocessing_dir(data_dir), f'token_counts.csv') 

############
### LOAD ###
############

def parse(path):
    with open(path, 'r') as f:
        for l in f:
            yield json.loads(l)

def load_business_data(data_dir):
    keys = ['business_id', 'city', 'state', 'categories']
    df = {}
    for k in keys:
        df[k] = []
    with open(os.path.join(raw_data_dir(data_dir), 'yelp_academic_dataset_business.json'), 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            for k in keys:
                df[k].append(data[k])
    business_df = pd.DataFrame(df)
    return business_df

#####################
### PREPROCESSING ###
#####################

def compute_token_length(data_dir):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    token_counts = []
    with open(os.path.join(raw_data_dir(data_dir), 'yelp_academic_dataset_review.json'), 'r') as f:
        text_list = []
        for i, line in enumerate(f):
            if i % 100000==0:
                print(f'Processed {i} reviews')
            data = json.loads(line)
            text = data['text']
            text_list.append(text)
            if len(text_list)==1024:
                tokens = tokenizer(text_list,
                                   padding='do_not_pad',
                                   truncation='do_not_truncate',
                                   return_token_type_ids=False,
                                   return_attention_mask=False,
                                   return_overflowing_tokens=False,
                                   return_special_tokens_mask=False,
                                   return_offsets_mapping=False,
                                   return_length=True)
                token_counts += tokens['length']
                text_list = []
        if len(text_list)>0:
            tokens = tokenizer(text_list,
                               padding='do_not_pad',
                               truncation='do_not_truncate',
                               return_token_type_ids=False,
                               return_attention_mask=False,
                               return_overflowing_tokens=False,
                               return_special_tokens_mask=False,
                               return_offsets_mapping=False,
                               return_length=True)
            token_counts += tokens['length']

    csv_path = token_length_path(data_dir) 
    df = pd.DataFrame({'token_counts': token_counts})
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

def process_reviews(data_dir):
    # load pre-computed token length
    assert os.path.exists(token_length_path(data_dir)), 'pre-compute token length first'
    token_length = pd.read_csv(token_length_path(data_dir))['token_counts'].values

    # filter and export
    with open(reviews_path(data_dir), 'w') as f:
        fields = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']
        writer = csv.DictWriter(f, fields, quoting=csv.QUOTE_NONNUMERIC)

        for i, review in enumerate(parse(os.path.join(raw_data_dir(data_dir), 'yelp_academic_dataset_review.json'))):
            if 'text' not in review:
                continue
            if len(review['text'].strip())==0:
                continue
            if token_length[i] > 512:
                continue
            row = {}
            for field in fields:
                row[field] = review[field]
            writer.writerow(row)
    # compute year
    df = pd.read_csv(reviews_path(data_dir), names=fields,
            dtype={'review_id': str, 'user_id': str, 'business_id':str, 'stars': int, 
                   'useful': int, 'funny': int, 'cool':int, 'text': str, 'date':str},
            keep_default_na=False, na_values=[])
    print(f'Before deduplication: {df.shape}')
    df['year'] = df['date'].apply(lambda x: int(x.split('-')[0]))
    # remove duplicates
    duplicated_within_user = df[['user_id','text']].duplicated()
    df_deduplicated_within_user = df[~duplicated_within_user]
    duplicated_text = df_deduplicated_within_user[df_deduplicated_within_user['text'].apply(lambda x: x.lower()).duplicated(keep=False)]['text']
    duplicated_text = set(duplicated_text.values)
    if len(duplicated_text)>0:
        print('Eliminating reviews with the following duplicate texts:')
        print('\n'.join(list(duplicated_text)))
        print('')
    df['duplicate'] = ((df['text'].isin(duplicated_text)) | duplicated_within_user)
    df = df[~df['duplicate']]
    print(f'After deduplication: {df[~df["duplicate"]].shape}')
    business_df = load_business_data(data_dir)
    df = pd.merge(df, business_df, on='business_id', how='left')
    df = df.drop(columns=['duplicate'])
    df.to_csv(reviews_path(data_dir), index=False, quoting=csv.QUOTE_NONNUMERIC)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    args = parser.parse_args()

    for dirpath in [splits_dir(data_dir(args.root_dir)), preprocessing_dir(data_dir(args.root_dir))]:
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
   
    compute_token_length(data_dir(args.root_dir))
    process_reviews(data_dir(args.root_dir))

if __name__=='__main__':
    main()    
