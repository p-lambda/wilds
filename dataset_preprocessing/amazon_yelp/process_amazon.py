import os, json, gzip, argparse, time, csv, urllib
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.core import k_core
from transformers import AutoTokenizer, BertTokenizerFast, BertTokenizer
from utils import *

CATEGORIES = ["AMAZON_FASHION", "All_Beauty","Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books", "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games"]

#############
### PATHS ###
#############

def data_dir(root_dir):
    return os.path.join(root_dir, 'amazon', 'data')

def raw_reviews_path(data_dir, category):
    return os.path.join(raw_data_dir(data_dir), category+'.json.gz')

def user_list_path(data_dir):
    return os.path.join(preprocessing_dir(data_dir), f'users.txt')

def product_list_path(data_dir):
    return os.path.join(preprocessing_dir(data_dir), f'products.txt')

def token_length_dir(data_dir):
    return os.path.join(preprocessing_dir(data_dir), 'token_length')

def token_length_path(data_dir, category):
    return os.path.join(token_length_dir(data_dir), f'{category}.csv')

def reviews_with_duplicates_path(data_dir):
    return os.path.join(preprocessing_dir(data_dir), f'reviews_with_duplicates.csv')

###############
### LOADING ###
###############

def download(data_dir, category):
    url = f'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/{category}_5.json.gz'
    print(url)
    urllib.request.urlretrieve(url, raw_reviews_path(data_dir, category))

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

#####################
### PREPROCESSING ###
#####################

def compute_token_length(data_dir, category):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    token_counts = []
    text_list = []
    print(category)        
    for i, review in enumerate(parse(raw_reviews_path(data_dir, category))):
        if 'reviewText' not in review or len(review['reviewText'].strip())==0:
            text = ""
        else:
            text = review['reviewText']
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
    csv_path =token_length_path(data_dir, category)
    df = pd.DataFrame({'token_counts': token_counts})
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

def process_k_core(data_dir, k):
    product_nodes = set()
    user_nodes = set()
    # compute k core products and users
    graph = nx.Graph()
    for category in CATEGORIES:
        print(category)
        token_length = pd.read_csv(token_length_path(data_dir, category))['token_counts'].values
        for i, review in enumerate(parse(raw_reviews_path(data_dir, category))):
            if 'reviewText' not in review:
                continue
            if len(review['reviewText'].strip())==0:
                continue
            if token_length[i] > 512:
                continue
            product_id = review['asin']
            user_id = review['reviewerID']
            if product_id not in product_nodes:
                graph.add_node(product_id, is_product=True)
                product_nodes.add(product_id)
            if user_id not in user_nodes:
                graph.add_node(user_id, is_product=False)
                user_nodes.add(user_id)
            graph.add_edge(user_id, product_id)
        assert token_length.size==(i+1), f'{token_length.size}, {i}'

    k_core_graph = k_core(graph, k=k)
    k_core_nodes = set(k_core_graph.nodes)
    with open(user_list_path(data_dir), 'w') as f_user:
        with open(product_list_path(data_dir), 'w') as f_product:
            for node in k_core_graph.nodes:
                assert not (node in product_nodes and node in user_nodes)
                if node in product_nodes:
                    f_product.write(f'{node}\n')
                elif node in user_nodes:
                    f_user.write(f'{node}\n')
    # load k core products and users
    print('loading users and product IDs...')
    user_df = pd.read_csv(user_list_path(data_dir), names=['user_id'])
    user_ids = set(user_df['user_id'])
    product_df = pd.read_csv(product_list_path(data_dir), names=['product_id'])
    product_ids = set(product_df['product_id'])
    # save reviews in k-core subset
    with open(reviews_with_duplicates_path(data_dir), 'w') as f:
        field_list = ['reviewerID','asin','overall','reviewTime','unixReviewTime','reviewText','summary','verified','category']
        writer = csv.DictWriter(f, field_list, quoting=csv.QUOTE_NONNUMERIC)
        for category in CATEGORIES:
            print(category) 
            token_length = pd.read_csv(token_length_path(data_dir, category))['token_counts'].values
            for i, review in enumerate(parse(raw_reviews_path(data_dir, category))):
                if 'reviewText' not in review:
                    continue
                if len(review['reviewText'].strip())==0:
                    continue
                if token_length[i] > 512:
                    continue
                product_id = review['asin']
                user_id = review['reviewerID']
                if user_id in user_ids and product_id in product_ids:
                    row = {}
                    for field in field_list:
                        if field=='category':
                            row[field] = category
                        elif field in review:
                            row[field] = review[field]
                        else:
                            print(f'missing {field}')
                            row[field] = ""
                    writer.writerow(row)
    # remove duplicates
    df = pd.read_csv(reviews_with_duplicates_path(data_dir), names=field_list,
                     dtype={'reviewerID':str, 'asin':str, 'reviewTime':str,'unixReviewTime':int,
                            'reviewText':str,'summary':str,'verified':bool,'category':str},
                     keep_default_na=False, na_values=[])
    df['reviewYear'] = df['reviewTime'].apply(lambda x: int(x.split(',')[-1]))
    df = df.drop_duplicates(['asin', 'reviewerID', 'overall', 'reviewTime'])
    df.to_csv(reviews_path(data_dir), index=False, quoting=csv.QUOTE_NONNUMERIC)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--k_core', type=int, default=30)
    args = parser.parse_args()
   
    for dirpath in [splits_dir(data_dir(args.root_dir)), preprocessing_dir(data_dir(args.root_dir)), token_length_dir(data_dir(args.root_dir))]:
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

    for category in CATEGORIES:
        download(data_dir(args.root_dir), category)
    for category in CATEGORIES:
        compute_token_length(data_dir(args.root_dir), category)
    process_k_core(data_dir(args.root_dir), args.k_core)

if __name__=='__main__':
    main()
