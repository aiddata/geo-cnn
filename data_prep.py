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

print('-' * 40)

print("\nInitializing...")


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

# cat_names = ['low', 'med', 'high']
cat_names = [0, 1, 2]

ncats = len(cat_names)

# ntl classes
ntl_class_bins = {
    0: [0, 3],
    1: [3, 8],
    2: [8, 63]
}

# ntl year
ntl_year = 2010


# data types to subset
type_names = ["train", "val", "test", "predict"]

# ratio for data types (must sum to 1.0)
type_weights = [0.850, 0.150, 0.0, 0.0]


# grid settings
overwrite_grid = False
pixel_size = 0.12
nfill = 400


# overwrtie class/type definitions
overwrite_full = False

# boundary path definiing grid area
tza_adm0_path = os.path.join(base_path, 'data/TZA_ADM0_GADM28_simplified.geojson')

# raw grid
grid_path = os.path.join(
    base_path,
    "data/sample_grid_{}.csv".format(str(pixel_size).split(".")[1])
)

# full set of data (before trimming)
full_path = os.path.join(
    base_path,
    "data/sample_grid_{}_full.csv".format(str(pixel_size).split(".")[1])
)

# -----------------------------------------------------------------------------


class NTL():
    def __init__(self, calibrated=False):
        self.base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites"
        if calibrated:
            self.base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"
    def set_year(self, year):
        self.year = year
        self.path = glob.glob(os.path.join(self.base, "*{0}*.tif".format(self.year)))[0]
        self.file = rasterio.open(self.path)
    def value(self, lon, lat, ntl_dim=7):
        """Get nighttime lights average value for grid around point
        """
        r, c = self.file.index(lon, lat)
        ntl_win = ((r-ntl_dim/2+1, r+ntl_dim/2+1), (c-ntl_dim/2+1, c+ntl_dim/2+1))
        ntl_data = self.file.read(1, window=ntl_win)
        ntl_mean = ntl_data.mean()
        return ntl_mean


def gen_sample_size(count, weights):
    """Given a `count` of total samples and list of weights,
    determine the number of samples associated with each weight
    """
    type_sizes = np.zeros(len(weights)).astype(int)
    # note: sizes are based on nkeep which is for a single NTL class label
    # subsequent steps to label data class (train, val, etc) must be repeated
    # for each NTL class
    for i,x in enumerate(weights):
        type_sizes[i] = int(np.floor(count * x))
    # used floor for all counts, so total can be short by one
    active_type_weights_indexes = [i for i,j in enumerate(weights) if j>0]
    for _ in range(count - sum(type_sizes)):
        bonus_type_weights_index = active_type_weights_indexes[np.random.randint(0, len(active_type_weights_indexes))]
        type_sizes[bonus_type_weights_index] += 1
    return type_sizes


def apply_types(data, classes, names, weights):
    """For each subset of data based on class,
    assign data type names based on given weights.
    Weight values must sum to one and all names must
    be assigned a weight.
    """
    for c in classes:
        cat_size = len(data.loc[data['label'] == c])
        # example of alternative method (simpler, but does no ensure consistent class sizes)
        # data.loc[data['label'] == c, 'type'] = np.random.choice(type_names, size=(cat_size,), p=type_weights)
        type_sizes = gen_sample_size(cat_size, weights)
        type_list = [[x] * type_sizes[i] for i,x in enumerate(names)]
        type_list = list(itertools.chain.from_iterable(type_list))
        np.random.shuffle(type_list)
        data.loc[data['label'] == c, 'type'] = type_list
    return data


def normalize(data, type_field, type_values, class_field, class_values):
    """
    create equal class sizes based on smallest class size
        - randomizes which extras from larger classes are dropped
    """
    for j in type_values:
        tmp_data = data.loc[data[type_field] == j].copy(deep=True)
        raw_class_sizes = [sum(tmp_data[class_field] == i) for i in class_values]
        nkeep = min(raw_class_sizes)
        tmp_data['drop'] = 'drop'
        for i in class_values:
            class_index = tmp_data.loc[tmp_data[class_field] == i].index
            keepers = np.random.permutation(class_index)[:nkeep]
            data.loc[keepers, 'drop'] = 'keep'
    data = data.loc[data['drop'] == 'keep'].copy(deep=True)
    return data


# -----------------------------------------------------------------------------


print("\nPreparing grid..")

# ntl data
ntl = NTL(calibrated=False)
ntl.set_year(ntl_year)

# boundary
tza_adm0 = fiona.open(tza_adm0_path)

# define, build, and save sample grid
if not os.path.isfile(grid_path) or overwrite_grid:
    grid = PointGrid(tza_adm0)
    grid.grid(pixel_size)
    # geo_path = os.path.join(base_path, "data/sample_grid.geojson")
    # grid.to_geojson(geo_path)
    # grid_path = os.path.join(base_path, "data/sample_grid.csv")
    # grid.to_csv(grid_path)
    grid.df = grid.to_dataframe()
    grid.gfill(nfill)
    grid.to_csv(grid_path)

df = pd.read_csv(grid_path, sep=",", encoding='utf-8')

# -----------------------------------------------------------------------------

print("\nBuilding datasets...")

# copy of full df

original_df = df.copy(deep=True)

# for use in debugging
# df = original_df.copy(deep=True)


# =====================================


if not os.path.isfile(full_path) or overwrite_full:
    # get ntl values
    df['ntl'] = df.apply(lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=7), axis=1)
    # label each point based on ntl value for point and class bins
    df["label"] = None
    for c, b in ntl_class_bins.iteritems():
        df.loc[df['ntl'] >= b[0], 'label'] = int(c)
    # ----------------------------------------
    # determine size of each data type (train, val, etc)
    df['type'] = None
    # subset to original grid (no spatial overlap)
    tmp_df = df.loc[df["group"] == "orig"].copy(deep=True)
    # define data group type
    data = apply_types(tmp_df, cat_names, type_names, type_weights)
    # based on classes for original grid subset, apply classes to
    # all associated subgrid points
    for i, row in tmp_df.iterrows():
        cell_id = row["cell_id"]
        df.loc[df["cell_id"] == cell_id, 'type'] = row['type']
    # save full set of data
    df.to_csv(full_path, index=False, encoding='utf-8')


# -----------------------------------------------------------------------------


df = pd.read_csv(full_path, sep=",", encoding='utf-8')

# print resulting split of classes for each data type
print("Full data:")
for i in type_names:
    tmp_df = df.loc[df['type'] == i]
    print("Samples per cat ({}):".format(i))
    for j in cat_names: print("{0}: {1}".format(j, sum(tmp_df['label'] == j)))


print("\nNormalizing class sizes...")

ndf = normalize(df, 'type', type_names, 'label', cat_names)

print("Normalized data:")
for i in type_names:
    tmp_df = ndf.loc[ndf['type'] == i]
    print("Samples per cat ({}):".format(i))
    for j in cat_names: print("{0}: {1}".format(j, sum(tmp_df['label'] == j)))


# -----------------------------------------------------------------------------


dataframe_dict = {}

print("Final data:")
for i in type_names:
    dataframe_dict[i] = ndf.loc[ndf['type'] == i]
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
