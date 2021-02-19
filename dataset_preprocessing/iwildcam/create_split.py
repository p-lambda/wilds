
import pandas as pd
import numpy as np

# Examples to skip due to e.g them missing, loading issues
LOCATIONS_TO_SKIP = [537]

CANNOT_OPEN = ['99136aa6-21bc-11ea-a13a-137349068a90.jpg',
               '87022118-21bc-11ea-a13a-137349068a90.jpg',
               '8f17b296-21bc-11ea-a13a-137349068a90.jpg',
               '883572ba-21bc-11ea-a13a-137349068a90.jpg',
               '896c1198-21bc-11ea-a13a-137349068a90.jpg',
               '8792549a-21bc-11ea-a13a-137349068a90.jpg',
               '94529be0-21bc-11ea-a13a-137349068a90.jpg']

CANNOT_LOAD = ['929da9de-21bc-11ea-a13a-137349068a90.jpg',
               '9631e6a0-21bc-11ea-a13a-137349068a90.jpg',
               '8c3a31fc-21bc-11ea-a13a-137349068a90.jpg',
               '88313344-21bc-11ea-a13a-137349068a90.jpg',
               '8c53e822-21bc-11ea-a13a-137349068a90.jpg',
               '911848a8-21bc-11ea-a13a-137349068a90.jpg',
               '98bd006c-21bc-11ea-a13a-137349068a90.jpg',
               '91ba7b50-21bc-11ea-a13a-137349068a90.jpg',
               '9799f64a-21bc-11ea-a13a-137349068a90.jpg',
               '88007592-21bc-11ea-a13a-137349068a90.jpg',
               '94860606-21bc-11ea-a13a-137349068a90.jpg',
               '9166fbd8-21bc-11ea-a13a-137349068a90.jpg']

OTHER = ['8e0c091a-21bc-11ea-a13a-137349068a90.jpg'] # This one got slightly different error


IDS_TO_SKIP = CANNOT_OPEN + CANNOT_LOAD + OTHER


def create_split(data_dir):
    train_df, val_cis_df, val_trans_df, test_cis_df, test_trans_df = _create_split(data_dir, seed=0)

    train_df.to_csv(data_dir / 'train.csv')
    val_cis_df.to_csv(data_dir / 'val_cis.csv')
    val_trans_df.to_csv(data_dir / 'val_trans.csv')
    test_cis_df.to_csv(data_dir / 'test_cis.csv')
    test_trans_df.to_csv(data_dir / 'test_trans.csv')


def _create_split(data_dir, seed, skip=True):
    data_dir = Path(data_dir)
    np_rng = np.random.default_rng(seed)

    # Load Kaggle train data
    with open(data_dir / r'iwildcam2020_train_annotations.json' ) as json_file:
        data = json.load(json_file)

    # This line was adapted from
    # https://www.kaggle.com/ateplyuk/iwildcam2020-pytorch-start
    df = pd.DataFrame(
            {
                'id': [item['id'] for item in data['annotations']],
                'category_id': [item['category_id'] for item in data['annotations']],
                'image_id': [item['image_id'] for item in data['annotations']],
                'location': [item['location'] for item in data['images']],
                'filename': [item['file_name'] for item in data['images']],
                'datetime': [item['datetime'] for item in data['images']],
                'frame_num': [item['frame_num'] for item in data['images']], # this attribute is not used
                'seq_id': [item['seq_id'] for item in data['images']] # this attribute is not used
            })


    # Extract the date from the datetime.
    df['datetime_obj'] = df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df['date'] = df['datetime_obj'].apply(lambda x: x.date())

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
    frac_validation = 0.05
    frac_test = 0.05
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

    # Remove examples that are corrupted in some way
    if skip:
        train_df, val_cis_df, val_trans_df, test_cis_df, test_trans_df = remove([train_df, val_cis_df,
                                                                                val_trans_df, test_cis_df,
                                                                                test_trans_df])


    # Reset index
    train_df.reset_index(inplace=True), val_cis_df.reset_index(inplace=True), val_trans_df.reset_index(inplace=True)
    test_cis_df.reset_index(inplace=True), test_trans_df.reset_index(inplace=True)

    # Make sure there's no overlap
    for df in [val_cis_df, val_trans_df, test_cis_df, test_trans_df]:
        assert not check_overlap(train_df, df)

    return train_df, val_cis_df, val_trans_df, test_cis_df, test_trans_df

def remove(dfs):
    new_dfs = []
    for df in dfs:
        df = df[~df['location'].isin(LOCATIONS_TO_SKIP)]
        df = df[~df['filename'].isin(IDS_TO_SKIP)]
        new_dfs.append(df)
    return new_dfs

def check_overlap(df1, df2):
    files1 = set(df1['filename'])
    files2 = set(df2['filename'])
    intersection = files1.intersection(files2)
    n_intersection = len(intersection)

    return False if n_intersection == 0 else True
