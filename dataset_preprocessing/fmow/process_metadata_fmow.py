from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from wilds.datasets.fmow_dataset import categories
from PIL import Image
import shutil
import time

root = Path('/u/scr/nlp/dro/fMoW/')
dstroot = Path('/u/scr/nlp/dro/fMoW/data')

# build test and seq mapping

with open(root / 'test_gt_mapping.json', 'r') as f:
    test_mapping = json.load(f)
with open(root / 'seq_gt_mapping.json', 'r') as f:
    seq_mapping = json.load(f)

def process_mapping(mapping):
    new_mapping = {}
    for pair in tqdm(mapping):
        new_mapping[pair['input']] = pair['output'] 
    return new_mapping

test_mapping = process_mapping(test_mapping)
seq_mapping = process_mapping(seq_mapping)


rgb_metadata = []
msrgb_metadata = []

for split in ['train', 'val', 'test', 'seq']:
    split_dir = root / (split + '_gt')

    len_split_dir = len(list(split_dir.iterdir()))
    for class_dir in tqdm(split_dir.iterdir(), total=len_split_dir):
        classname = class_dir.stem
        len_class_dir = len(list(class_dir.iterdir()))
        for class_subdir in tqdm(class_dir.iterdir(), total=len_class_dir):
            for metadata_file in class_subdir.iterdir():
                if metadata_file.suffix == '.json':
                    with open(metadata_file, 'r') as f:
                        metadata_json = json.load(f)

                    locs = metadata_json['raw_location'].split('((')[1].split('))')[0].split(',')
                    locs = [loc.strip().split(' ') for loc in locs]
                    locs = [[float(loc[0]), float(loc[1])] for loc in locs]
                    # lat long are reversed in locs
                    lats = [loc[1] for loc in locs]
                    lons = [loc[0] for loc in locs]

                    if split in {'train', 'val'}:
                        img_path = f"{split}/{metadata_file.parent.parent.stem}/{metadata_file.parent.stem}/{metadata_file.stem}.jpg"
                    else:
                        test_mapping_key = f"{split_dir.stem}/{metadata_file.parent.parent.stem}/{metadata_file.parent.stem}"
                        if split == 'test':
                            img_path_dir = Path(test_mapping[test_mapping_key])
                        else:
                            img_path_dir = Path(seq_mapping[test_mapping_key])

                        new_img_filename = metadata_file.stem.replace(str(metadata_file.parent.stem), img_path_dir.stem) + ".jpg"
                        img_path = img_path_dir / new_img_filename

                    curr_metadata = {
                        'split': split,
                        'img_filename': metadata_json['img_filename'],
                        'img_path': str(img_path),
                        'spatial_reference': metadata_json['spatial_reference'],
                        'epsg': metadata_json['epsg'],
                        'category': metadata_json['bounding_boxes'][1]['category'],
                        'visible': metadata_json['bounding_boxes'][1]['visible'],
                        'img_width': metadata_json['img_width'],
                        'img_height': metadata_json['img_height'],
                        'country_code': metadata_json['country_code'],
                        'cloud_cover': metadata_json['cloud_cover'],
                        'timestamp': metadata_json['timestamp'],
                        'lat': np.mean(lats),
                        'lon': np.mean(lons)}

                    if str(metadata_file).endswith('msrgb.json'):
                        msrgb_metadata.append(curr_metadata)
                    elif str(metadata_file).endswith('rgb.json'):
                        rgb_metadata.append(curr_metadata)


rgb_df = pd.DataFrame(rgb_metadata)
msrgb_df = pd.DataFrame(msrgb_metadata)

# add region
def add_region(df):
    country_codes_df = pd.read_csv(dstroot / 'country_code_mapping.csv')
    countrycode_to_region = {k: v for k, v in zip(country_codes_df['alpha-3'], country_codes_df['region'])}
    country_codes = df['country_code'].to_list()
    regions = [countrycode_to_region.get(code, 'Other') for code in country_codes]
    df['region'] = regions

add_region(rgb_df)
add_region(msrgb_df)

rgb_df.to_csv(dstroot / 'rgb_metadata.csv', index=False)
msrgb_df.to_csv(dstroot / 'msrgb_metadata.csv', index=False)

################ save rgb imgs to npy

category_to_idx = {cat: i for i, cat in enumerate(categories)}
default_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224)])
metadata = pd.read_csv(dstroot / 'rgb_metadata.csv')

num_batches = 100
batch_size = len(metadata) // num_batches
if len(metadata) % num_batches != 0:
    num_batches += 1

print("Saving into chunks...")
for j in tqdm(range(num_batches)):
    batch_metadata = metadata.iloc[j*batch_size : (j+1)*batch_size]
    imgs = []

    for i in tqdm(range(len(batch_metadata))):
        curr_metadata = batch_metadata.iloc[i].to_dict()

        img_path = root / curr_metadata['img_path']
        img = Image.open(img_path)
        img = img.convert('RGB')

        img = np.asarray(default_transform(img), dtype=np.uint8)

        imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.uint8)
    np.save(dstroot / f'rgb_all_imgs_{j}.npy', imgs)
