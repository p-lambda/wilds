import os, json, gzip, argparse, time, csv 
import numpy as np
import pandas as pd
from utils import *

CATEGORIES = ["AMAZON_FASHION", "All_Beauty","Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books", "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games"]

#############
### PATHS ###
#############

def data_dir(root_dir):
    return os.path.join(root_dir, 'amazon', 'data')

def generate_user_splits(data_dir, reviews_df, min_size_per_user,
                         train_size, eval_size, seed):
    # mark duplicates
    duplicated_within_user = reviews_df[['reviewerID','reviewText']].duplicated()
    df_deduplicated_within_user = reviews_df[~duplicated_within_user]
    duplicated_text = df_deduplicated_within_user[df_deduplicated_within_user['reviewText'].apply(lambda x: x.lower()).duplicated(keep=False)]['reviewText']
    duplicated_text = set(duplicated_text.values)
    reviews_df['duplicate'] = ((reviews_df['reviewText'].isin(duplicated_text)) | duplicated_within_user)
    # mark html candidates
    reviews_df['contains_html'] = reviews_df['reviewText'].apply(lambda x: '<' in x and '>' in x)
    # mark clean ones
    reviews_df['clean'] = (~reviews_df['duplicate'] & ~reviews_df['contains_html'])

    # generate splits
    generate_group_splits(
            data_dir=data_dir,
            reviews_df=reviews_df,
            min_size_per_group=min_size_per_user, 
            group_field='reviewerID',
            split_name='user',
            train_size=train_size,
            eval_size=eval_size,
            seed=seed,
            select_column='clean')

def generate_users_baseline_splits(data_dir, reviews_df, reviewer_id, seed, user_split_name='user'):
    # seed
    np.random.seed(seed)
    # sizes
    n, _ = reviews_df.shape
    splits = np.ones(n)*-1
    # load user split
    orig_splits_df = pd.read_csv(splits_path(data_dir, user_split_name))
    splits[((orig_splits_df['split']==OOD_TEST) & (reviews_df['reviewerID']==reviewer_id)).values] = TEST
    # train
    train_indices, = np.where(np.logical_and.reduce((reviews_df['reviewerID']==reviewer_id,
                                                     splits==-1,
                                                     orig_splits_df['clean'])))
    np.random.shuffle(train_indices)
    eval_size = np.sum(splits==TEST)
    splits[train_indices[:eval_size]] = VAL
    splits[train_indices[eval_size:]] = TRAIN
    split_df = pd.DataFrame({'split': splits})
    split_df.to_csv(splits_path(data_dir, f'{reviewer_id}_baseline'), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    args = parser.parse_args()

    df = pd.read_csv(reviews_path(data_dir(args.root_dir)),
                     dtype={'reviewerID':str, 'asin':str, 'reviewTime':str,'unixReviewTime':int,
                            'reviewText':str,'summary':str,'verified':bool,'category':str, 'reviewYear':int},
                     keep_default_na=False, na_values=[])

    # category subpopulation
    generate_fixed_group_splits(
            data_dir=data_dir(args.root_dir),
            reviews_df=df,
            group_field='category', 
            train_groups=None,
            split_name='category_subpopulation', 
            train_size=int(1e6), 
            eval_size_per_group=1000, 
            seed=0)

    # category generalization and baselines
    train_categories_list = [[category,] for category in CATEGORIES] + \
            [['Books','Movies_and_TV','Home_and_Kitchen','Electronics'],
             ['Movies_and_TV','Books'],
             ['Movies_and_TV','Books','Home_and_Kitchen']]
    for train_categories in train_categories_list:
        split_name = ','.join([category.lower() for category in train_categories])+'_generalization'
        generate_fixed_group_splits(
                data_dir=data_dir(args.root_dir),
                reviews_df=df,
                group_field='category', 
                train_groups=train_categories,
                split_name=split_name, 
                train_size=int(1e6), 
                eval_size_per_group=1000, 
                seed=0)
                        
    # time shift
    generate_time_splits(
            data_dir=data_dir(args.root_dir),
            reviews_df=df,
            year_field='reviewYear',
            year_threshold=2013,
            train_size=int(1e6),
            eval_size_per_year=4000,
            seed=0)

    # user splits 
    generate_user_splits(
            data_dir=data_dir(args.root_dir),
            reviews_df=df,
            min_size_per_user=150,
            train_size=int(1e6), 
            eval_size=1e5,
            seed=0)

    baseline_reviewers = ['AV6QDP8Q0ONK4', 'A37BRR2L8PX3R2', 'A1UH21GLZTYYR5', 'ASVY5XSYJ1XOE', 'A1NE43T0OM6NNX',
                          'A9Q28YTLYREO7', 'A1CNQTCRQ35IMM', 'A20EEWWSFMZ1PN', 'A3JVZY05VLMYEM', 'A219Y76LD1VP4N']
    for reviewer_id in baseline_reviewers:
        generate_users_baseline_splits(
		data_dir=data_dir(args.root_dir),
                reviews_df=df, 
                reviewer_id=reviewer_id,
                seed=0)

if __name__=='__main__':
    main()
