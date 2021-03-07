from datetime import datetime
from pathlib import Path
import argparse
import json
from PIL import Image

import pandas as pd
import numpy as np

def create_split(data_dir, seed):
    np_rng = np.random.default_rng(seed)

    # Loading json was adapted from
    # https://www.kaggle.com/ateplyuk/iwildcam2020-pytorch-start
    filename = f'iwildcam2021_train_annotations_final.json'
    with open(data_dir / filename ) as json_file:
        data = json.load(json_file)

    df_annotations = pd.DataFrame({
         'category_id': [item['category_id'] for item in data['annotations']],
         'image_id': [item['image_id'] for item in data['annotations']]
    })

    df_metadata = pd.DataFrame({
          'image_id': [item['id'] for item in data['images']],
          'location': [item['location'] for item in data['images']],
          'filename': [item['file_name'] for item in data['images']],
          'datetime': [item['datetime'] for item in data['images']],
          'frame_num': [item['frame_num'] for item in data['images']], # this attribute is not used
          'seq_id': [item['seq_id'] for item in data['images']] # this attribute is not used
      })


    df = df_metadata.merge(df_annotations, on='image_id', how='inner')

    # Create category_id to name dictionary
    cat_id_to_name_map = {}
    for item in data['categories']:
        cat_id_to_name_map[item['id']] = item['name']
    df['category_name'] = df['category_id'].apply(lambda x: cat_id_to_name_map[x])

    # Extract the date from the datetime.
    df['datetime_obj'] = df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df['date'] = df['datetime_obj'].apply(lambda x: x.date())

    # Retrieve the sequences that span 2 days
    grouped_by = df.groupby('seq_id')
    nunique_dates = grouped_by['date'].nunique()
    seq_ids_that_span_across_days = nunique_dates[nunique_dates.values > 1].reset_index()['seq_id'].values

    # Split by location to get the cis & trans validation set
    locations = np.unique(df['location'])
    n_locations = len(locations)
    frac_val_locations = 0.10
    frac_test_locations = 0.15
    n_val_locations = int(frac_val_locations * n_locations)
    n_test_locations = int(frac_test_locations * n_locations)
    n_train_locations = n_locations - n_val_locations - n_test_locations

    np_rng.shuffle(locations) # Shuffle, then split
    train_locations, val_trans_locations = locations[:n_train_locations], locations[n_train_locations:(n_train_locations+n_val_locations)]
    test_trans_locations = locations[(n_train_locations+n_val_locations):]


    remaining_df, val_trans_df = df[df['location'].isin(train_locations)], df[df['location'].isin(val_trans_locations)]
    test_trans_df = df[df['location'].isin(test_trans_locations)]

    # Split remaining samples by dates to get the cis validation and test set
    frac_validation = 0.07
    frac_test = 0.09
    unique_dates = np.unique(remaining_df['date'])
    n_dates = len(unique_dates)
    n_val_dates = int(n_dates * frac_validation)
    n_test_dates = int(n_dates * frac_test)
    n_train_dates = n_dates - n_val_dates - n_test_dates

    np_rng.shuffle(unique_dates) # Shuffle, then split
    train_dates, val_cis_dates = unique_dates[:n_train_dates], unique_dates[n_train_dates:(n_train_dates+n_val_dates)]
    test_cis_dates = unique_dates[(n_train_dates+n_val_dates):]

    val_cis_df = remaining_df[remaining_df['date'].isin(val_cis_dates)]
    test_cis_df = remaining_df[remaining_df['date'].isin(test_cis_dates)]
    train_df = remaining_df[remaining_df['date'].isin(train_dates)]

    # Locations in val_cis and test_cis but not in train are all moved to train set
    # since we want all locations in tcis splits to be in the train set.
    locs_to_be_moved = []
    locs_to_be_moved.extend(list(set(val_cis_df['location']) - set(train_df['location'])))
    locs_to_be_moved.extend(list(set(test_cis_df['location']) - set(train_df['location'])))

    df_to_be_moved = []
    df_to_be_moved.append(val_cis_df[val_cis_df['location'].isin(locs_to_be_moved)])
    df_to_be_moved.append(test_cis_df[test_cis_df['location'].isin(locs_to_be_moved)])
    df_to_be_moved = pd.concat(df_to_be_moved)
    train_df = pd.concat([train_df, df_to_be_moved])

    val_cis_df = val_cis_df[~val_cis_df['location'].isin(locs_to_be_moved)]
    test_cis_df = test_cis_df[~test_cis_df['location'].isin(locs_to_be_moved)]

    # Remove examples from test with classes that are not in train
    train_classes = set(train_df['category_id'].unique())
    val_cis_df = val_cis_df[val_cis_df['category_id'].isin(train_classes)]
    val_trans_df = val_trans_df[val_trans_df['category_id'].isin(train_classes)]
    test_cis_df = test_cis_df[test_cis_df['category_id'].isin(train_classes)]
    test_trans_df = test_trans_df[test_trans_df['category_id'].isin(train_classes)]

    # Assert that all sequences that spanned across multiple days ended up in the same split
    for seq_id in seq_ids_that_span_across_days:
        n_splits = 0
        for split_df in [train_df, val_cis_df, test_cis_df]:
            if seq_id in split_df['seq_id'].values:
                n_splits += 1
            assert n_splits == 1, "Each sequence should only be in one split. Please move manually"

    # Reset index
    train_df.reset_index(inplace=True, drop=True), val_cis_df.reset_index(inplace=True, drop=True), val_trans_df.reset_index(inplace=True, drop=True)
    test_cis_df.reset_index(inplace=True, drop=True), test_trans_df.reset_index(inplace=True, drop=True)

    print("n train: ", len(train_df))
    print("n val trans: ", len(val_trans_df))
    print("n test trans: ", len(test_trans_df))
    print("n val cis: ", len(val_cis_df))
    print("n test cis: ", len(test_cis_df))

    # Merge into one df
    train_df['split'] = 'train'
    val_trans_df['split'] = 'val'
    test_trans_df['split'] = 'test'
    val_cis_df['split'] = 'id_val'
    test_cis_df['split'] = 'id_test'
    df = pd.concat([train_df, val_trans_df, test_trans_df, test_cis_df, val_cis_df])
    df = df.reset_index(drop=True)

    # Create y labels by remapping the category ids to be contiguous
    unique_categories = np.unique(df['category_id'])
    n_classes = len(unique_categories)
    category_to_label = dict([(i, j) for i, j in zip(unique_categories, range(n_classes))])
    df['y'] = df['category_id'].apply(lambda x: category_to_label[x]).values
    print("N classes: ", n_classes)

    # Create y to category name map and save
    categories_df = pd.DataFrame({
          'category_id': [item['id'] for item in data['categories']],
          'name': [item['name'] for item in data['categories']]
      })

    categories_df['y'] = categories_df['category_id'].apply(lambda x: category_to_label[x] if x in category_to_label else 99999)
    categories_df = categories_df.sort_values('y').reset_index(drop=True)
    categories_df = categories_df[['y','category_id','name']]

    # Create remapped location id such that they are contigious
    location_ids = df['location']
    locations = np.unique(location_ids)
    n_groups = len(locations)
    location_to_group_id = {locations[i]: i for i in range(n_groups)}
    df['location_remapped' ] = df['location'].apply(lambda x: location_to_group_id[x])

    # Create remapped sequence id such that they are contigious
    sequence_ids = df['seq_id']
    sequences = np.unique(sequence_ids)
    n_sequences = len(sequences)
    sequence_to_normalized_id = {sequences[i]: i for i in range(n_sequences)}
    df['sequence_remapped' ] = df['seq_id'].apply(lambda x: sequence_to_normalized_id[x])


    # Make sure there's no overlap
    for split_df in [val_cis_df, val_trans_df, test_cis_df, test_trans_df]:
        assert not check_overlap(train_df, split_df)

    # Save
    df = df.sort_values(['split','location_remapped', 'sequence_remapped','datetime']).reset_index(drop=True)
    cols = ['split', 'location_remapped', 'location', 'sequence_remapped', 'seq_id',  'y', 'category_id', 'datetime', 'filename', 'image_id']
    df[cols].to_csv(data_dir / 'metadata.csv')
    categories_df.to_csv(data_dir / 'categories.csv', index=False)


def check_overlap(df1, df2, column='filename'):
    files1 = set(df1[column])
    files2 = set(df2[column])
    intersection = files1.intersection(files2)
    n_intersection = len(intersection)

    return False if n_intersection == 0 else True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()

    create_split(Path(args.data_dir), seed=0)
