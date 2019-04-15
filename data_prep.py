from __future__ import print_function, division

import os
import copy
import itertools

import pandas as pd
import numpy as np
import fiona

from create_grid import PointGrid
from load_data import NTL_Reader


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



class PrepareSamples():

    def __init__(self, base_path, static_params):

        # -----------------
        # output settings

        overwrite_all = False

        self.overwrite_grid = False | overwrite_all # base grid generation
        self.overwrite_full = False | overwrite_all # overwrite class/type definitions
        self.overwrite_trim = False | overwrite_all # overwrite class size trimming (which is randomized)


        # grid_tag = "{}_{}".format(str(pixel_size).split(".")[1], "cal")
        # grid_tag = "{}_{}".format(str(pixel_size).split(".")[1], "raw")
        grid_tag = str(static_params["grid_pixel_size"]).split(".")[1]

        # raw grid
        self.grid_path = os.path.join(
            base_path,
            "data/grid/sample_grid_init_{}.csv".format(grid_tag)
        )

        # full set of data (before trimming)
        self.full_path = os.path.join(
            base_path,
            "data/grid/sample_grid_data_{}.csv".format(grid_tag)
        )

        # final set of data (after trimming)
        self.trim_path = os.path.join(
            base_path,
            "data/grid/sample_grid_trim_{}.csv".format(grid_tag)
        )

        # -----------------

        # boundary path defining grid area
        self.boundary_path = static_params["boundary_path"]

        # -----------------
        # grid settings

        # base grid resolution
        self.pixel_size = static_params["grid_pixel_size"]
        # number of additional points to fill for each base point
        self.nfill = static_params["grid_nfill"]
        # distance from base point to fill in
        self.fill_dist = static_params["grid_fill_dist"]
        # fixed or random fill of point
        self.fill_mode = static_params["grid_fill_mode"]

        # -----------------
        # ntl settings

        # ntl classes (starting value for each bin, ends at following value)
        #   First value should always be 0
        #   Final value capped at max of data
        self.ntl_class_bins = static_params["ntl_class_bins"]

        # whether to use calibrated ntl data or original
        self.ntl_calibrated = static_params["ntl_calibrated"]

        # ntl year
        self.ntl_year = static_params["ntl_year"]

        # size of square (pixels) to calculate ntl
        self.ntl_dim = static_params["ntl_dim"]

        # -----------------

        # ntl cateogories (low, med, high)
        #   must match number of values in ntl_class_bins
        self.cat_names = static_params["cat_names"]

        # data types to subset
        self.type_names = static_params["type_names"]

        # ratio for data types (must sum to 1.0)
        self.type_weights = static_params["type_weights"]


    def prepare_grid(self):
        # load boundary data
        boundary_src = fiona.open(self.boundary_path)
        grid = PointGrid(boundary_src)
        boundary_src.close()
        grid.grid(self.pixel_size)
        # geo_path = os.path.join(base_path, "data/sample_grid.geojson")
        # grid.to_geojson(geo_path)
        # grid_path = os.path.join(base_path, "data/sample_grid.csv")
        # grid.to_csv(grid_path)
        grid.df = grid.to_dataframe()
        grid.gfill(self.nfill, distance=self.fill_dist, mode=self.fill_mode)
        grid.to_csv(self.grid_path)
        self.grid_df = grid.df.copy(deep=True)


    def build_datasets(self):
        # ntl data
        self.ntl = NTL_Reader(calibrated=self.ntl_calibrated)
        self.ntl.set_year(self.ntl_year)
        # ----------
        df = self.grid_df.copy(deep=True)
        # get ntl values
        df['ntl'] = df.apply(lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)
        # label each point based on ntl value for point and class bins
        df["label"] = None
        for c, b in enumerate(self.ntl_class_bins):
            df.loc[df['ntl'] >= b, 'label'] = int(c)
        # ----------------------------------------
        # determine size of each data type (train, val, etc)
        df['type'] = None
        # subset to original grid (no spatial overlap)
        tmp_df = df.loc[df["group"] == "orig"].copy(deep=True)
        # define data group type
        self.df = apply_types(tmp_df, self.cat_names, self.type_names, self.type_weights)
        # based on classes for original grid subset, apply classes to
        # all associated subgrid points
        for _, row in self.df.iterrows():
            cell_id = row["cell_id"]
            df.loc[df["cell_id"] == cell_id, 'type'] = row['type']
        # save full set of data
        self.df.to_csv(self.full_path, index=False, encoding='utf-8')


    def normalize_classes(self):
        self.ndf = normalize(self.df, 'type', self.type_names, 'label', self.cat_names)
        self.ndf.to_csv(self.trim_path, index=False, encoding='utf-8')


    def run(self):

        print("\nPreparing grid..")
        # define, build, and save sample grid
        if not os.path.isfile(self.grid_path) or self.overwrite_grid:
            self.prepare_grid()
        else:
            self.grid_df = pd.read_csv(self.grid_path, sep=",", encoding='utf-8')


        print("\nBuilding datasets...")
        if not os.path.isfile(self.full_path) or self.overwrite_full:
            self.build_datasets()
        else:
            self.df = pd.read_csv(self.full_path, sep=",", encoding='utf-8')


        print("\nNormalizing class sizes...")
        if not os.path.isfile(self.trim_path) or self.overwrite_trim:
            self.normalize_classes()
        else:
            self.ndf = pd.read_csv(self.trim_path, sep=",", encoding='utf-8')


        print("\nPreparing dataframe dict...")

        dataframe_dict = {}
        for i in self.type_names:
            dataframe_dict[i] = self.ndf.loc[self.ndf['type'] == i]

        class_sizes = {
            j:[sum(dataframe_dict[j]['label'] == i) for i in self.cat_names] for j in self.type_names
        }

        return dataframe_dict, class_sizes


    def print_counts(self):

        # print resulting split of classes for each data type
        print("Full data:")
        for i in self.type_names:
            tmp_df = self.df.loc[self.df['type'] == i]
            print("Samples per cat ({}):".format(i))
            for j in self.cat_names: print("{0}: {1}".format(j, sum(tmp_df['label'] == j)))


        print("Normalized data:")
        for i in self.type_names:
            tmp_df = self.ndf.loc[self.ndf['type'] == i]
            print("Samples per cat ({}):".format(i))
            for j in self.cat_names: print("{0}: {1}".format(j, sum(tmp_df['label'] == j)))
