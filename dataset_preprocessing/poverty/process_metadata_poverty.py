########
# ADAPTED from github.com/sustainlab-group/africa_poverty
########

import tensorflow as tf
import numpy as np
import batcher
import dataset_constants
from tqdm import tqdm
from utils.general import load_npz
import pickle
import pandas as pd
from pathlib import Path


FOLDS = ['A', 'B', 'C', 'D', 'E']
SPLITS = ['train', 'val', 'test']
BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
DATASET = '2009-17'
ROOT = Path('../data') # Path to files from sustainlab-group/africa_poverty
DSTROOT = Path('/u/scr/nlp/dro/poverty/data')

COUNTRIES = np.asarray(dataset_constants.DHS_COUNTRIES)

file_path = ROOT / 'dhs_image_hists.npz'
npz = load_npz(file_path)

labels = npz['labels']
locs = npz['locs']
years = npz['years']
nls_center = npz['nls_center']
nls_mean = npz['nls_mean']

num_examples = len(labels)
assert np.all(np.asarray([len(labels), len(locs), len(years)]) == num_examples)

dmsp_mask = years < 2012
viirs_mask = ~dmsp_mask

with open(ROOT / 'dhs_loc_dict.pkl', 'rb') as f:
    loc_dict = pickle.load(f)

df_data = []
for label, loc, nl_mean, nl_center in zip(labels, locs, nls_mean, nls_center):
    lat, lon = loc
    loc_info = loc_dict[(lat, lon)]
    country = loc_info['country']
    year = int(loc_info['country_year'][-4:])  # use the year matching the surveyID
    urban = loc_info['urban']
    household = loc_info['households']
    row = [lat, lon, label, country, year, urban, nl_mean, nl_center, household]
    df_data.append(row)
df = pd.DataFrame.from_records(
    df_data,
    columns=['lat', 'lon', 'wealthpooled', 'country', 'year', 'urban', 'nl_mean', 'nl_center', 'households'])

df.to_csv(DSTROOT / 'dhs_metadata.csv', index=False)
