from __future__ import print_function, division

import os
import copy
import itertools
import errno

import pandas as pd
import numpy as np
import fiona

from create_grid import PointGrid, SampleFill
from load_ntl_data import NTL_Reader


def make_dirs(path_list):
    for path in path_list:
        make_dir(path)

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


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

    def __init__(self, base_path, static_params, version, overwrite=False):

        self.base_path = base_path
        self.static_params = static_params
        self.version = version

        # -----------------
        # sample settings

        self.overwrite = overwrite


        self.sample_type = static_params["sample_type"]
        if self.sample_type not in ["source", "grid"]:
            raise ValueError("sample type must be either `source` or `grid`")

        # init = raw sample data
        # fill = raw sample data filled with additional sample points (grouped with original points)
        # ntl  = ntl values added
        # full = full set of data (before trimming)
        # trim = final set of data (after trimming)
        sample_stages = ["init", "fill", "ntl", "full", "trim"]


        if self.sample_type == "source":
            tag_name = static_params["source_name"]
        elif self.sample_type == "grid":
            tag_name = str(static_params["grid_pixel_size"]).split(".")[1]

        if static_params["sample_fill_dist"] < 0:
            raise ValueError("Sample fill dist must be greater than or equal to zero (Given: {})".format(static_params["sample_fill_dist"]))



        sample_tag = "{}_{}_{}_{}_{}".format(
            tag_name,
            static_params["sample_nfill"],
            str(static_params["sample_fill_dist"]).split(".")[1] if static_params["sample_fill_dist"] > 0 else "0",
            static_params["sample_fill_mode"],
            version
        )

        self.sample_path = {}
        for i in sample_stages:
            self.sample_path[i] = os.path.join(
                base_path,
                "data/grid/sample_{}_{}_{}.csv".format(self.sample_type, i, sample_tag)
            )

        make_dir(os.path.join(base_path, "data/sample"))


        # -----------------

        # boundary path defining grid area
        self.boundary_path = static_params["boundary_path"]


        # -----------------
        # sample fill  settings

        # number of additional points to fill for each base point
        self.nfill = static_params["sample_nfill"]
        # distance from base point to fill in
        self.fill_dist = static_params["sample_fill_dist"]
        # fixed or random fill of point
        self.fill_mode = static_params["sample_fill_mode"]

        # -----------------
        # ntl settings

        # dmsp or viirs
        self.ntl_type = static_params["ntl_type"]

        # whether to use calibrated ntl data or original
        self.ntl_calibrated = static_params["ntl_calibrated"]

        # ntl year
        self.ntl_year = static_params["ntl_year"]

        # size of square (pixels) to calculate ntl
        self.ntl_dim = static_params["ntl_dim"]

        # minimum NTL to keep (for dropping zero/low values that may be noise)
        self.ntl_min = static_params["ntl_min"]

        # -----------------

        # field name used to define cat values
        self.cat_field = static_params["cat_field"]

        # starting value for each bin, ends at (not including) following value
        self.cat_bins = static_params["cat_bins"]

        # number of items in cat_names must match number of items in cat_bins
        self.cat_names = static_params["cat_names"]

        # data types to subset
        self.type_names = static_params["type_names"]

        # ratio for data types (must sum to 1.0)
        self.type_weights = static_params["type_weights"]


    def prepare_sample(self):
        if self.sample_type == "source":
            self._prepare_source_sample()
        elif self.sample_type == "grid":
            self._prepare_grid_sample()


    def _prepare_source_sample(self):
        df = pd.read_csv(self.static_params["source_path"], sep=",", encoding='utf-8')
        cols = df.columns
        if "lon" not in cols and "longitude" in cols:
            df["lon"] = df.longitude
        if "lat" not in cols and "latitude" in cols:
            df["lat"] = df.latitude
        if "lon" not in df.columns and "lat" not in df.columsn:
            raise Exception("Source for sample data must contain longitude/latitude or lon/lat columns")
        df["sample_id"] = range(len(df))
        self.sample_df = df.copy(deep=True)


    def _prepare_grid_sample(self):
        # load boundary data
        boundary_src = fiona.open(self.boundary_path)
        grid = PointGrid(boundary_src)
        boundary_src.close()
        grid.grid(self.static_params["grid_pixel_size"])
        # geo_path = os.path.join(base_path, "data/sample_grid.geojson")
        # grid.to_geojson(geo_path)
        # grid_init_path = os.path.join(base_path, "data/sample_grid.csv")
        # grid.to_csv(grid_init_path)
        grid.df = grid.to_dataframe()
        grid.to_csv(self.sample_path["init"])
        self.sample_df = grid.df.copy(deep=True)


    def fill_sample(self):
        fill = SampleFill(self.sample_df)
        fill.gfill(self.nfill, distance=self.fill_dist, mode=self.fill_mode)
        self.sample_df = fill.df.copy(deep=True)
        self.sample_df.to_csv(self.sample_path["fill"])


    def assign_ntl(self):
        # ntl data
        self.ntl = NTL_Reader(ntl_type=self.ntl_type, calibrated=self.ntl_calibrated)
        self.ntl.set_year(self.ntl_year)
        # ----------
        self.df = self.sample_df.copy(deep=True)
        # get ntl values
        self.df['ntl'] = self.df.apply(lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)
        self.df = self.df.loc[self.df['ntl'] >= self.ntl_min]
        self.df.to_csv(self.sample_path["ntl"], index=False, encoding='utf-8')


    def build_datasets(self):
        # label each point based on cat ntl value for point and class bins
        self.df["label"] = None
        for c, b in enumerate(self.cat_bins):
            self.df.loc[self.df[self.cat_field] >= b, 'label'] = int(c)
        # ----------------------------------------
        # determine size of each data type (train, val, etc)
        self.df['type'] = None
        # subset to original grid (no spatial overlap)
        tmp_df = self.df.loc[self.df["group"] == "orig"].copy(deep=True)
        # define data group type for original grid
        tmp_df = apply_types(tmp_df, self.cat_names, self.type_names, self.type_weights)
        # based on classes for original grid subset, apply classes to
        # all associated subgrid points
        for _, row in tmp_df.iterrows():
            sample_id = row["sample_id"]
            self.df.loc[self.df["sample_id"] == sample_id, 'type'] = row['type']
        # save full set of data
        self.df.to_csv(self.sample_path["full"], index=False, encoding='utf-8')


    def normalize_classes(self):
        self.ndf = normalize(self.df, 'type', self.type_names, 'label', self.cat_names)
        self.ndf.to_csv(self.sample_path["trim"], index=False, encoding='utf-8')


    def run(self):

        print("\nPreparing grid...")
        # define, build, and save sample grid
        if not os.path.isfile(self.sample_path["init"]) or self.overwrite:
            self.prepare_sample()
        else:
            self.sample_df = pd.read_csv(self.sample_path["init"], sep=",", encoding='utf-8')

        print("\nFilling in sample...")
        if not os.path.isfile(self.sample_path["fill"]) or self.overwrite:
            self.fill_sample()
        else:
            self.sample_df = pd.read_csv(self.sample_path["fill"], sep=",", encoding='utf-8')

        print("\nAdding NTL values...")
        if not os.path.isfile(self.sample_path["ntl"]) or self.overwrite:
            self.assign_ntl()
        else:
            self.df = pd.read_csv(self.sample_path["ntl"], sep=",", encoding='utf-8')

        print("\nBuilding datasets...")
        if not os.path.isfile(self.sample_path["full"]) or self.overwrite:
            self.build_datasets()
        else:
            self.df = pd.read_csv(self.sample_path["full"], sep=",", encoding='utf-8')

        print("\nNormalizing class sizes...")
        if not os.path.isfile(self.sample_path["trim"]) or self.overwrite:
            self.normalize_classes()
        else:
            self.ndf = pd.read_csv(self.sample_path["trim"], sep=",", encoding='utf-8')


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
        print("\nFull data:")
        for i in self.type_names:
            tmp_df = self.df.loc[self.df['type'] == i]
            print("\tSamples per cat ({}):".format(i))
            for j in self.cat_names: print("\t{0}: {1}".format(j, sum(tmp_df['label'] == j)))


        print("\nNormalized data:")
        for i in self.type_names:
            tmp_df = self.ndf.loc[self.ndf['type'] == i]
            print("\tSamples per cat ({}):".format(i))
            for j in self.cat_names: print("\t{0}: {1}".format(j, sum(tmp_df['label'] == j)))
