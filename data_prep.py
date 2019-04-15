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
from load_data import NTL_Reader
from settings_builder import Settings

print('-' * 40)

print("\nInitializing...")


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

json_path = "settings_example.json"

s = Settings(base_path)
s.load(json_path)
s.set_param_count()
tasks = s.hashed_iter()


# boundary path defining grid area
boundary_path = s.static["boundary_path"]

# -----------------
# grid settings

# base grid resolution
pixel_size = s.static["pixel_size"]
# number of additional points to fill for each base point
nfill = s.static["nfill"]
# distance from base point to fill in
fill_dist = s.static["fill_dist"]
# fixed or random fill of point
fill_mode = s.static["fill_mode"]

# -----------------
# ntl settings

# ntl classes (starting value for each bin, ends at following value)
#   First value should always be 0
#   Final value capped at max of data
ntl_class_bins = s.static["ntl_class_bins"]

# whether to use calibrated ntl data or original
ntl_calibrated = s.static["ntl_calibrated"]

# ntl year
ntl_year = s.static["ntl_year"]

# size of square (pixels) to calculate ntl
ntl_dim = s.static["ntl_dim"]

# -----------------

# ntl cateogories (low, med, high)
#   must match number of values in ntl_class_bins
cat_names = s.static["cat_names"]

# data types to subset
type_names = s.static["type_names"]

# ratio for data types (must sum to 1.0)
type_weights = s.static["type_weights"]


# -----------------
# output settings

overwrite_all = False

overwrite_grid = False | overwrite_all # base grid generation
overwrite_full = False | overwrite_all # overwrite class/type definitions
overwrite_trim = False | overwrite_all # overwrite class size trimming (which is randomized)


# grid_tag = "{}_{}".format(str(pixel_size).split(".")[1], "cal")
# grid_tag = "{}_{}".format(str(pixel_size).split(".")[1], "raw")
grid_tag = str(pixel_size).split(".")[1]

# raw grid
grid_path = os.path.join(
    base_path,
    "data/grid/sample_grid_init_{}.csv".format(grid_tag)
)

# full set of data (before trimming)
full_path = os.path.join(
    base_path,
    "data/grid/sample_grid_data_{}.csv".format(grid_tag)
)

# final set of data (after trimming)
trim_path = os.path.join(
    base_path,
    "data/grid/sample_grid_trim_{}.csv".format(grid_tag)
)


# -----------------------------------------------------------------------------


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

# load boundary data
boundary_src = fiona.open(boundary_path)

# define, build, and save sample grid
if not os.path.isfile(grid_path) or overwrite_grid:
    grid = PointGrid(boundary_src)
    grid.grid(pixel_size)
    # geo_path = os.path.join(base_path, "data/sample_grid.geojson")
    # grid.to_geojson(geo_path)
    # grid_path = os.path.join(base_path, "data/sample_grid.csv")
    # grid.to_csv(grid_path)
    grid.df = grid.to_dataframe()
    grid.gfill(nfill, distance=fill_dist, mode=fill_mode)
    grid.to_csv(grid_path)

grid_df = pd.read_csv(grid_path, sep=",", encoding='utf-8')


# -----------------------------------------------------------------------------


print("\nBuilding datasets...")

# ntl data
ntl = NTL_Reader(calibrated=ntl_calibrated)
ntl.set_year(ntl_year)

if not os.path.isfile(full_path) or overwrite_full:
    df = grid_df.copy(deep=True)
    # get ntl values
    df['ntl'] = df.apply(lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=ntl_dim), axis=1)
    # label each point based on ntl value for point and class bins
    df["label"] = None
    for c, b in enumerate(ntl_class_bins):
        df.loc[df['ntl'] >= b, 'label'] = int(c)
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
else:
    df = pd.read_csv(full_path, sep=",", encoding='utf-8')


print("\nNormalizing class sizes...")

if not os.path.isfile(trim_path) or overwrite_trim:
    ndf = normalize(df, 'type', type_names, 'label', cat_names)
    ndf.to_csv(trim_path, index=False, encoding='utf-8')

else:
    ndf = pd.read_csv(trim_path, sep=",", encoding='utf-8')


print("\nPreparing dataframe dict...")

dataframe_dict = {}
for i in type_names:
    dataframe_dict[i] = ndf.loc[ndf['type'] == i]


# -----------------------------------------------------------------------------


# print resulting split of classes for each data type
print("Full data:")
for i in type_names:
    tmp_df = df.loc[df['type'] == i]
    print("Samples per cat ({}):".format(i))
    for j in cat_names: print("{0}: {1}".format(j, sum(tmp_df['label'] == j)))


print("Normalized data:")
for i in type_names:
    tmp_df = ndf.loc[ndf['type'] == i]
    print("Samples per cat ({}):".format(i))
    for j in cat_names: print("{0}: {1}".format(j, sum(tmp_df['label'] == j)))


# -----------------------------------------------------------------------------


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
