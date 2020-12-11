import os, json, gzip, argparse, time, csv 
import numpy as np
import pandas as pd
from utils import *

def data_dir(root_dir):
    return os.path.join(root_dir, 'yelp', 'data')

def load_reviews(data_dir):
    reviews_df = pd.read_csv(reviews_path(data_dir),
                    dtype={'review_id': str, 'user_id': str, 'business_id':str, 'stars': int, 
                           'useful': int, 'funny': int, 'cool':int, 'text': str, 'date':str},
                    keep_default_na=False, na_values=[])
    return reviews_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    args = parser.parse_args()

    reviews_df = load_reviews(data_dir(args.root_dir))
    # time
    generate_time_splits(
            data_dir=data_dir(args.root_dir),
            reviews_df=reviews_df,
            year_field='year',
            year_threshold=2013,
            train_size=int(1e6),
            eval_size_per_year=1000,
            seed=0)

    # user shifts
    generate_group_splits(
            data_dir=data_dir(args.root_dir),
            reviews_df=reviews_df, 
            min_size_per_group=50, 
            group_field='user_id',
            split_name='user',
            train_size=int(1e6),
            eval_size=int(4e4),
            seed=0)

if __name__=='__main__':
    main()
