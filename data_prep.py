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


# cat_names = ['low', 'med', 'high']
cat_names = [0, 1, 2]

ncats = len(cat_names)

base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"


# -----------------------------------------------------------------------------


class NTL():
    def __init__(self):
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


ntl_year = 2010
ntl = NTL()
ntl.set_year(ntl_year)


# -----------------------------------------------------------------------------


print("\nPreparing grid..")

# boundary
tza_adm0_path = os.path.join(base_path, 'data/TZA_ADM0_GADM28_simplified.geojson')
tza_adm0 = fiona.open(tza_adm0_path)


# define, build, and save sample grid
pixel_size = 0.008

overwrite_grid = False
# pixel_size = 0.06
# print(pixel_size)

csv_path = os.path.join(
    base_path,
    "data/sample_grid_{}.csv".format(str(pixel_size).split(".")[1])
)

if os.path.isfile(csv_path) and not overwrite_grid:
    df = pd.read_csv(csv_path, sep=",", encoding='utf-8')
else:
    grid = PointGrid(tza_adm0)
    grid.grid(pixel_size)
    # geo_path = os.path.join(base_path, "data/sample_grid.geojson")
    # grid.to_geojson(geo_path)
    # csv_path = os.path.join(base_path, "data/sample_grid.csv")
    # grid.to_csv(csv_path)
    grid.df = grid.to_dataframe()
    # look up ntl values for each grid cell
    grid.df['ntl'] = grid.df.apply(lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=7), axis=1)
    grid.to_csv(csv_path)
    df = grid.df.copy(deep=True)

# ==============================
# copy of full df

original_df = df.copy(deep=True)

# for use in debugging
# df = original_df.copy(deep=True)


# -----------------------------------------------------------------------------


print("\nBuilding datasets...")


# label each point based on ntl value for point and class bins

class_bins = {
    0: [0, 3],
    1: [3, 10],
    2: [9, 63]
}

df["label"] = None
for c, b in class_bins.iteritems():
    df.loc[df['ntl'] >= b[0], 'label'] = int(c)


# ==============================

# determine size of each data type (train, val, etc)

type_names = ["train", "val", "test", "predict"]

# ratio for train, val, test data
# must sum to 1.0
type_weights = [0.850, 0.150, 0.0, 0.0]


def gen_sample_size(count, weights):
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


# ==============================


df['type'] = None

# repeat for each NTL class (cat_names)
for c in cat_names:
    cat_size = len(df.loc[df['label'] == c])
    # example of alternative method (simpler, but does no ensure consistent class sizes)
    # df.loc[df['label'] == c, 'type'] = np.random.choice(type_names, size=(cat_size,), p=type_weights)
    type_sizes = gen_sample_size(cat_size, type_weights)
    type_list = [[x] * type_sizes[i] for i,x in enumerate(type_names)]
    type_list = list(itertools.chain.from_iterable(type_list))
    np.random.shuffle(type_list)
    df.loc[df['label'] == c, 'type'] = type_list





for i in type_names:
    tmp_df = df.loc[df['type'] == i]
    print("Samples per cat ({}):".format(i))
    for j in cat_names: print("{0}: {1}".format(j, sum(tmp_df['label'] == j)))



# -----------------------------------------------------------------------------

print("\nNormalizing class sizes...")

# create equal class sizes based on smallest class size
#   - randomizes which extras from larger classes are dropped

raw_class_sizes = [sum(df['label'] == i) for i in cat_names]

nkeep = min(raw_class_sizes)

df['drop'] = 'drop'

for i in cat_names:
    class_index = df.loc[df['label'] == i].index
    n_keep = min(raw_class_sizes)
    keepers = np.random.permutation(class_index)[:nkeep]
    df.loc[keepers, 'drop'] = 'keep'

df = df.loc[df['drop'] == 'keep'].copy(deep=True)



print("Samples per cat (raw):")
for i in cat_names: print("{0}: {1}".format(i, sum(original_df['label'] == i)))

print("Samples per cat (reduced):")
for i in cat_names: print("{0}: {1}".format(i, sum(df['label'] == i)))



# -----------------------------------------------------------------------------


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
