import pandas as pd
from matplotlib import pyplot as plt
import os,sys
import numpy as np
from tqdm import tqdm
import argparse

from attr_definitions import GROUP_ATTRS, AGGREGATE_ATTRS, ORIG_ATTRS

def load_df(root):
    """
    Loads the data and removes all examples where we don't have identity annotations.
    """
    df = pd.read_csv(os.path.join(root, 'all_data.csv'))
    df = df.loc[(df['identity_annotator_count'] > 0), :]
    df = df.reset_index(drop=True)
    return df

def augment_df(df):
    """
    Augment the dataframe with auxiliary attributes.
    First, we create aggregate attributes, like `LGBTQ` or `other_religions`.
    These are aggregated because there would otherwise not be enough examples to accurately
    estimate their accuracy.

    Next, for each category of demographics (e.g., race, gender), we construct an auxiliary
    attribute (e.g., `na_race`, `na_gender`) that is 1 if the comment has no identities related to
    that demographic, and is 0 otherwise.
    Note that we can't just create a single multi-valued attribute like `gender` because there's
    substantial overlap: for example, 4.6% of comments mention both male and female identities.
    """
    df = df.copy()
    for aggregate_attr in AGGREGATE_ATTRS:
        aggregate_mask = pd.Series([False] * len(df))
        for attr in AGGREGATE_ATTRS[aggregate_attr]:
            attr_mask = (df[attr] >= 0.5)
            aggregate_mask = aggregate_mask | attr_mask
        df[aggregate_attr] = 0
        df.loc[aggregate_mask, aggregate_attr] = 1

    attr_count = np.zeros(len(df))
    for attr in ORIG_ATTRS:
        attr_mask = (df[attr] >= 0.5)
        attr_count += attr_mask
    df['num_identities'] = attr_count
    df['more_than_one_identity'] = (attr_count > 1)

    for group in GROUP_ATTRS:
        print(f'## {group}')
        counts = {}
        na_mask = np.ones(len(df))
        for attr in GROUP_ATTRS[group]:
            attr_mask = (df[attr] >= 0.5)
            na_mask = na_mask & ~attr_mask
            counts[attr] = np.mean(attr_mask)
        counts['n/a'] = np.mean(na_mask)

        col_name = f'na_{group}'
        df[col_name] = 0
        df.loc[na_mask, col_name] = 1

        for k, v in counts.items():
            print(f'{k:40s}: {v:.4f}')
        print()
    return df

def construct_splits(df):
    """
    Construct splits.
    The original data already has a train vs. test split.
    We triple the size of the test set so that we can better estimate accuracy on the small groups,
    and construct a validation set by randomly sampling articles.
    """

    df = df.copy()
    train_df = df.loc[df['split'] == 'train']
    test_df = df.loc[df['split'] == 'test']
    train_articles = set(train_df['article_id'].values)
    test_articles = set(test_df['article_id'].values)
    # Assert no overlap between train and test articles
    assert len(train_articles.intersection(test_articles)) == 0

    n_train = len(train_df)
    n_test = len(test_df)
    n_train_articles = len(train_articles)
    n_test_articles = len(test_articles)

    ## Set params
    n_val_articles = n_test_articles
    n_new_test_articles = 2 * n_test_articles

    np.random.seed(0)

    # Sample val articles
    val_articles = np.random.choice(
        list(train_articles),
        size=n_val_articles,
        replace=False)
    df.loc[df['article_id'].isin(val_articles), 'split'] = 'val'

    # Sample new test articles
    train_articles = train_articles - set(val_articles)
    new_test_articles = np.random.choice(
        list(train_articles),
        size=n_new_test_articles,
        replace=False)
    df.loc[df['article_id'].isin(new_test_articles), 'split'] = 'test'

    train_df = df.loc[df['split'] == 'train']
    val_df = df.loc[df['split'] == 'val']
    test_df = df.loc[df['split'] == 'test']

    train_articles = set(train_df['article_id'].values)
    val_articles = set(val_df['article_id'].values)
    test_articles = set(test_df['article_id'].values)

    # Sanity checks
    assert len(df) == len(train_df) + len(val_df) + len(test_df)
    assert n_train == len(train_df) + len(val_df) + np.sum(df['article_id'].isin(new_test_articles))
    assert n_test == len(test_df) - np.sum(df['article_id'].isin(new_test_articles))
    assert n_train_articles == len(train_articles) + len(val_articles) + len(new_test_articles)
    assert n_val_articles == len(val_articles)
    assert n_test_articles == len(test_articles) - n_new_test_articles
    assert len(train_articles.intersection(val_articles)) == 0
    assert len(train_articles.intersection(test_articles)) == 0
    assert len(val_articles.intersection(test_articles)) == 0

    print('% of examples')
    for split in ['train', 'val', 'test']:
        print(split, np.mean(df['split'] == split), np.sum(df['split'] == split))
    print('')

    print('class balance')
    for split in ['train', 'val', 'test']:
        split_df = df.loc[df['split'] == split]
        print('pos', np.mean(split_df['toxicity'] > 0.5))
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()

    df = load_df(args.root)
    df = augment_df(df)
    df = construct_splits(df)
    df.to_csv(os.path.join(args.root, f'all_data_with_identities.csv'))
