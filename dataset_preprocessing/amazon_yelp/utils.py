import os, json, gzip, argparse, time, csv 
import numpy as np
import pandas as pd

TRAIN, VAL, TEST = range(3)
_, OOD_VAL, ID_VAL, OOD_TEST, ID_TEST = range(5)

#############
### PATHS ###
#############

def raw_data_dir(data_dir):
    return os.path.join(data_dir, 'raw')

def preprocessing_dir(data_dir):
    return os.path.join(data_dir, 'preprocessing')

def splits_dir(data_dir):
    return os.path.join(data_dir, 'splits')

def reviews_path(data_dir):
    return os.path.join(data_dir, f'reviews.csv')

def splits_path(data_dir, split_name):
    return os.path.join(splits_dir(data_dir), f'{split_name}.csv')

##############
### SPLITS ###
##############

def generate_time_splits(data_dir, reviews_df, year_field, year_threshold, train_size, eval_size_per_year, seed):
    # seed
    np.random.seed(seed)
    # sizes
    n, _ = reviews_df.shape
    splits = np.ones(n)*-1
    baseline_splits = np.ones(n)*-1
    # val and test
    for year in range(min(reviews_df[year_field]), max(reviews_df[year_field])+1):
        year_indices, = np.where(reviews_df[year_field]==year)
        if year_indices.size==0:
            print(f"{year} is empty")
            continue
        if year_indices.size < eval_size_per_year*2:
            curr_eval_size = year_indices.size//2
        else:
            curr_eval_size = eval_size_per_year
        eval_indices = np.random.choice(year_indices, curr_eval_size*2,
                                        replace=False)
        if year <= year_threshold:
            splits[eval_indices[:curr_eval_size]] = ID_VAL 
            splits[eval_indices[curr_eval_size:]] = ID_TEST
        else:
            splits[eval_indices[:curr_eval_size]] = OOD_VAL 
            splits[eval_indices[curr_eval_size:]] = OOD_TEST
            baseline_splits[eval_indices[:curr_eval_size]] = VAL 
            baseline_splits[eval_indices[curr_eval_size:]] = TEST

    # train
    train_year_indices, = np.where(np.logical_and(reviews_df[year_field]<=year_threshold, splits==-1))
    train_indices = np.random.choice(train_year_indices, train_size, replace=False)
    splits[train_indices] = TRAIN
    baseline_train_year_indices, = np.where(np.logical_and(reviews_df[year_field]>year_threshold, splits==-1)) # require spits!=-1 to reserve ID_VAL, ID_TEST too
    baseline_train_indices = np.random.choice(baseline_train_year_indices, train_size, replace=False)
    baseline_splits[baseline_train_indices] = TRAIN
    # save
#    splits[np.where(splits==ID_TEST)] = -1 # Reserve but don't save ID TEST
    pd.DataFrame({'split': splits}).to_csv(splits_path(data_dir, 'time'), index=False)
    pd.DataFrame({'split': baseline_splits}).to_csv(splits_path(data_dir, 'time_baseline'), index=False)

def generate_group_splits(data_dir, reviews_df, min_size_per_group, group_field, split_name,
                          train_size, eval_size, seed, select_column=None):

    # seed
    np.random.seed(seed)
    # sizes
    n, _ = reviews_df.shape
    eval_size_per_group = min_size_per_group//2
    # get user IDs with sufficient user counts  
    if select_column is not None:
        group_counts = reviews_df[reviews_df[select_column]][group_field].value_counts().reset_index()
    else:
        group_counts = reviews_df[group_field].value_counts().reset_index()
    group_counts.columns = [group_field, 'count']
    group_counts.sort_values(group_field, ascending=False, inplace=True)
    groups = group_counts[group_counts['count']>=min_size_per_group][group_field].values
    np.random.shuffle(groups)
    print(groups)
    # initialize splits
    splits = np.ones(n)*-1
    # train and in-distribution eval
    group_idx = 0
    cumulative_train_size = 0
    cumulative_val_size = 0
    cumulative_test_size = 0
    while cumulative_train_size < train_size and group_idx<len(groups):
        curr_group = groups[group_idx]
        if select_column is not None:
            curr_group_indices, = np.where((reviews_df[group_field]==curr_group) & reviews_df[select_column])
        else:
            curr_group_indices, = np.where((reviews_df[group_field]==curr_group))
        curr_train_size = curr_group_indices.size - eval_size_per_group
        np.random.shuffle(curr_group_indices)
        splits[curr_group_indices[:curr_train_size]] = TRAIN
        if cumulative_val_size < eval_size:
            splits[curr_group_indices[curr_train_size:]] = ID_VAL
            cumulative_val_size += eval_size_per_group
        elif cumulative_test_size < eval_size:
            splits[curr_group_indices[curr_train_size:]] = ID_TEST
            cumulative_test_size += eval_size_per_group
        cumulative_train_size += curr_train_size
        group_idx += 1
    # unseen groups from the same distribution
    for split in (OOD_VAL, OOD_TEST):
        cumulative_eval_size = 0
        while cumulative_eval_size < eval_size and group_idx<len(groups):
            curr_group = groups[group_idx]
            if select_column is not None:
                curr_group_indices, = np.where((reviews_df[group_field]==curr_group) & reviews_df[select_column])
            else:
                curr_group_indices, = np.where((reviews_df[group_field]==curr_group))
            data_indices = np.random.choice(curr_group_indices,
                                            eval_size_per_group,
                                            replace=False)
            splits[data_indices] = split
            cumulative_eval_size += eval_size_per_group
            group_idx += 1

    if group_idx>=len(groups):
        print(f'ran out of groups for {outpath}')

    # save
#    splits[np.where(splits==ID_TEST)] = -1 # Reserve but don't save ID TEST
    df_dict = {'split': splits}
    if select_column is not None:
        df_dict[select_column] = reviews_df[select_column]
    split_df = pd.DataFrame(df_dict)
    split_df.to_csv(splits_path(data_dir, split_name), index=False)

def generate_fixed_group_splits(data_dir, reviews_df, group_field, train_groups, split_name, train_size, eval_size_per_group, seed):
    # seed
    np.random.seed(seed)
    # groups
    unique_groups = reviews_df[group_field].unique()
    # initialize
    n, _ = reviews_df.shape
    splits = np.ones(n)*-1
    # val and test
    for group in unique_groups:
        group_indices, = np.where(reviews_df[group_field]==group)
        if group_indices.size==0:
            print(f"{group} is empty")
            continue
        if group_indices.size < eval_size_per_group*2:
            curr_eval_size = group_indices.size//2
        else:
            curr_eval_size = eval_size_per_group
        eval_indices = np.random.choice(group_indices, curr_eval_size*2,
                                        replace=False)
        if train_groups is None: #subpopulation shift
            splits[eval_indices[:curr_eval_size]] = VAL
            splits[eval_indices[curr_eval_size:]] = TEST
        elif group in train_groups:
            splits[eval_indices[:curr_eval_size]] = ID_VAL
            splits[eval_indices[curr_eval_size:]] = ID_TEST
        else:
            splits[eval_indices[:curr_eval_size]] = OOD_VAL
            splits[eval_indices[curr_eval_size:]] = OOD_TEST
    # train
    if train_groups is None: #subpopulation shift
        train_group_indices, = np.where(splits==-1)
    else:
        train_group_indices, = np.where(np.logical_and(reviews_df[group_field].isin(set(train_groups)).ravel(), splits==-1))
    if train_group_indices.size <= train_size:
        train_indices = train_group_indices
    else:
        train_indices = np.random.choice(train_group_indices, train_size, replace=False)
    splits[train_indices] = TRAIN
    # save
#    splits[np.where(splits==ID_TEST)] = -1 # Reserve but don't save ID TEST
    split_df = pd.DataFrame({'split': splits})
    split_df.to_csv(splits_path(data_dir, split_name), index=False)
