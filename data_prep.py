from __future__ import print_function, division

import os
import copy
import glob
import itertools
import datetime
import time
import pprint

import rasterio
import pandas as pd
import numpy as np
import fiona

import torchvision
from torchvision import utils, datasets, models, transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import resnet

from create_grid import PointGrid


print("Initializing...")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Running on:", device)


run_types = {
    1: 'fine tuning',
    2: 'fixed feature extractor'
}


# cat_names = ['low', 'med', 'high']
cat_names = [0, 1, 2]

ncats = len(cat_names)


# -----------------------------------------------------------------------------


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

lsms_clusters_path = os.path.join(
    base_path, "survey_data/final/tanzania_lsms_cluster.csv")

lsms_cluster = pd.read_csv(lsms_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')


consumption_list = list(lsms_cluster['cons'])
cat_vals = [np.percentile(consumption_list, x*100/ncats) for x in range(1, ncats+1)]


def classify(val):
    for cix, cval in enumerate(cat_vals):
        if val <= cval:
            return cat_names[cix]
    print(val)
    raise Exception("Could not classify")

lsms_cluster['label'] = lsms_cluster.apply(
    lambda z: classify(z['cons']), axis=1)


# -----------------------------------------------------------------------------


ntl_base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"
ntl_year = 2010
ntl_path = glob.glob(os.path.join(ntl_base, "*{0}*.tif".format(ntl_year)))[0]
ntl_file = rasterio.open(ntl_path)

def get_ntl(lon, lat, ntl_dim=7):
    """Get nighttime lights average value for grid around point
    """
    r, c = ntl_file.index(lon, lat)
    ntl_win = ((r-ntl_dim/2+1, r+ntl_dim/2+1), (c-ntl_dim/2+1, c+ntl_dim/2+1))
    ntl_data = ntl_file.read(1, window=ntl_win)
    ntl_mean = ntl_data.mean()
    return ntl_mean


lsms_cluster['ntl'] = lsms_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)


class_ntl_means = dict(zip(cat_names, lsms_cluster.groupby('label')['ntl'].mean()))


# -----------------------------------------------------------------------------

print("Preparing grid")

# boundary
tza_adm0_path = os.path.join(base_path, 'run_data/TZA_ADM0_GADM28_simplified.geojson')
tza_adm0 = fiona.open(tza_adm0_path)


# define, build, and save sample grid
pixel_size = 0.008

csv_path = os.path.join(base_path, "run_data/sample_grid.csv")

if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path, sep=",", encoding='utf-8')
else:
    grid = PointGrid(tza_adm0)
    grid.grid(pixel_size)
    # geo_path = os.path.join(base_path, "run_data/sample_grid.geojson")
    # grid.to_geojson(geo_path)
    # csv_path = os.path.join(base_path, "run_data/sample_grid.csv")
    # grid.to_csv(csv_path)
    grid.df = grid.to_dataframe()
    # look up ntl values for each grid cell
    grid.df['ntl'] = grid.df.apply(lambda z: get_ntl(z['lon'], z['lat'], ntl_dim=7), axis=1)
    grid.to_csv(csv_path)
    df = grid.df.copy(deep=True)


# find nearest neighbor for each grid cell using class_ntl_means

# set all where ntl <  lowest class automatically to lowest class
df['label'] = None
df.loc[df['ntl'] <= class_ntl_means[min(cat_names)], 'label'] = min(cat_names)


def find_nn(val):
    nn_options = [(k, abs(val - v)) for k, v in class_ntl_means.iteritems()]
    nn_class = sorted(nn_options, key=lambda x: x[1])[0][0]
    return nn_class

# run nearest neigbor class label for remaining
to_label = df['ntl'] > class_ntl_means[min(cat_names)]
df.loc[to_label, 'label'] = df.loc[to_label].apply(lambda z: find_nn(z['ntl']), axis=1)



# drop out some of the extra from low category
original_df = df.copy(deep=True)

# for debugging
# df = original_df.copy(deep=True)



print("Samples per cat (raw):")
for i in cat_names: print("{0}: {1}".format(i, sum(df['label'] == i)))


# ==============================


# low_count = sum(df['label'] == 0)
# other_count = sum(df['label'] == 1) + sum(df['label'] == 2)

# keep_ratio = (other_count * 1.0) / low_count

# keep_ratio = keep_ratio if keep_ratio < 1 else 1

# df['drop'] = 'keep'
# df.loc[df['label'] == 0, 'drop'] = np.random.choice(["drop", "keep"], size=(low_count,), p=[1-keep_ratio, keep_ratio])


# ==============================
# equal class sizes based on smallest class size

raw_class_sizes = [sum(df['label'] == i) for i in cat_names]

nkeep = min(raw_class_sizes)

df['drop'] = 'drop'

for i in cat_names:
    class_index = df.loc[df['label'] == i].index
    n_keep = min(raw_class_sizes)
    keepers = np.random.permutation(class_index)[:nkeep]
    df.loc[keepers, 'drop'] = 'keep'


# ==============================


df = df.loc[df['drop'] == 'keep'].copy(deep=True)

print("Samples per cat (reduced):")
for i in cat_names: print("{0}: {1}".format(i, sum(df['label'] == i)))



# -----------------------------------------------------------------------------


print("Building datasets")


# ==========
# lsms_cluster['type'] = np.random.choice(["train", "val"], size=(len(lsms_cluster),), p=[0.90, 0.10])

# train_df = lsms_cluster.loc[lsms_cluster['type'] == "train"]
# val_df = lsms_cluster.loc[lsms_cluster['type'] == "val"]
# ==========


type_names = ["train", "val", "test", "predict"]
type_cats = range(len(type_names))

# ratio for train, val, test data
# must sum to 1.0
type_weights = [0.840, 0.150, 0.005, 0.005]
type_weights = [0.740, 0.150, 0.105, 0.005]

type_sizes = np.zeros(len(type_names)).astype(int)

# note: sizes are based on nkeep which is for a single NTL class label
# subsequent steps to label data class (train, val, etc) must be repeated
# for each NTL class
for i in type_cats:
    type_sizes[i] = int(np.floor(nkeep * type_weights[i]))

# used floor for all counts, so total can be short by one
for _ in range(nkeep-sum(type_sizes)):
    type_sizes[np.random.randint(0, len(type_names))] += 1


type_list = [[type_names[i]] * type_sizes[i] for i in type_cats]
type_list = list(itertools.chain.from_iterable(type_list))

df['type'] = None
# repeat for each NTL class (cat_names)
for i in cat_names:
    np.random.shuffle(type_list)
    df.loc[df['label'] == i, 'type'] = type_list





dataframe_dict = {}

for i in type_names:
    dataframe_dict[i] = df.loc[df['type'] == i]
    print("Samples per cat ({}):".format(i))
    for j in cat_names: print("{0}: {1}".format(j, sum(dataframe_dict[i]['label'] == j)))



# train_df = df.loc[df['type'] == "train"]
# val_df = df.loc[df['type'] == "val"]
# test_df = df.loc[df['type'] == "test"]
# predict_df = df.loc[df['type'] == "predict"]

# dataframe_dict = {
#     "train": train_df,
#     "val": val_df,
#     "test": test_df,
#     "predict": predict_df
# }


train_class_sizes = [sum(dataframe_dict["train"]['label'] == i) for i in cat_names]
val_class_sizes = [sum(dataframe_dict["val"]['label'] == i) for i in cat_names]
test_class_sizes = [sum(dataframe_dict["test"]['label'] == i) for i in cat_names]
predict_class_sizes = [sum(dataframe_dict["predict"]['label'] == i) for i in cat_names]


# print("Samples per cat (train):")
# for i in cat_names: print("{0}: {1}".format(i, sum(train_df['label'] == i)))

# print("Samples per cat (val):")
# for i in cat_names: print("{0}: {1}".format(i, sum(val_df['label'] == i)))

# print("Samples per cat (test):")
# for i in cat_names: print("{0}: {1}".format(i, sum(test_df['label'] == i)))

# print("Samples per cat (predict):")
# for i in cat_names: print("{0}: {1}".format(i, sum(predict_df['label'] == i)))


