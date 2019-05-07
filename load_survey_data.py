from __future__ import print_function, division

import os
import glob
import itertools
import rasterio
import pandas as pd
import numpy as np

from load_data import NTL_Reader


class SurveyData():
    """Load survey data based on static settings from settings json
    """

    def __init__(self, base_path, settings):

        self.base_path = base_path

        self.ntl_calibrated = settings["ntl_calibrated"]
        self.ntl_dim = settings["ntl_dim"]
        self.ntl_year = settings["ntl_year"]
        self.survey = settings["survey"]

        self.ntl = NTL_Reader(calibrated=self.ntl_calibrated)
        self.ntl.set_year(self.ntl_year)

        self.surveys = {}

        self._lsms2010_cluster()
        self._lsms2012_cluster()
        self._dhs2010_cluster()
        self._dhs2015_cluster()


    def duplicate(self, df):
        mod = 0.002
        df = df.drop("lonlat", axis=1)
        df["group"] = df.index
        new_df_dict = []
        for ix, row in df.iterrows():
            row_dict = row.to_dict()
            lon, lat = row_dict["lon"], row_dict["lat"]
            lon_vals = [lon - mod, lon, lon + mod]
            lat_vals = [lat + mod, lat, lat - mod]
            combos = itertools.product(lon_vals, lat_vals)
            for i in combos:
                tmp_row_dict = row_dict.copy()
                tmp_row_dict["lon"] = i[0]
                tmp_row_dict["lat"] = i[1]
                new_df_dict.append(tmp_row_dict)

        new_df = pd.DataFrame(new_df_dict)
        return new_df


    def _lsms2010_cluster(self):
        lsms2010_field = 'cons'

        lsms2010_clusters_path = os.path.join(
            self.base_path, "data/surveys/final/tanzania_2010_lsms_cluster.csv")

        lsms2010_cluster = pd.read_csv(lsms2010_clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        lsms2010_cluster = self.duplicate(lsms2010_cluster)

        lsms2010_cluster['ntl'] = lsms2010_cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        lsms2010_cluster["pred_yval"] = lsms2010_cluster[lsms2010_field]

        self.surveys["lsms2010_cluster"] = lsms2010_cluster


    def _lsms2012_cluster(self):
        lsms2012_field = 'cons'

        lsms2012_clusters_path = os.path.join(
            self.base_path, "data/surveys/final/tanzania_2012_lsms_cluster.csv")

        lsms2012_cluster = pd.read_csv(lsms2012_clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        lsms2012_cluster = self.duplicate(lsms2012_cluster)

        lsms2012_cluster['ntl'] = lsms2012_cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        lsms2012_cluster["pred_yval"] = lsms2012_cluster[lsms2012_field]

        self.surveys["lsms2012_cluster"] = lsms2012_cluster


    def _dhs2010_cluster(self):
        dhs2010_field = 'wealthscore'

        dhs2010_clusters_path = os.path.join(
            self.base_path, "data/surveys/final/tanzania_2010_dhs_cluster.csv")

        dhs2010_cluster = pd.read_csv(dhs2010_clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        dhs2010_cluster = self.duplicate(dhs2010_cluster)

        dhs2010_cluster['ntl'] = dhs2010_cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        dhs2010_cluster["pred_yval"] = dhs2010_cluster[dhs2010_field]

        self.surveys["dhs2010_cluster"] = dhs2010_cluster


    def _dhs2015_cluster(self):
        dhs2015_field = 'wealthscore'

        dhs2015_clusters_path = os.path.join(
            self.base_path, "data/surveys/final/tanzania_2015_dhs_cluster.csv")

        dhs2015_cluster = pd.read_csv(dhs2015_clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        dhs2015_cluster = self.duplicate(dhs2015_cluster)

        dhs2015_cluster['ntl'] = dhs2015_cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        dhs2015_cluster["pred_yval"] = dhs2015_cluster[dhs2015_field]

        self.surveys["dhs2015_cluster"] = dhs2015_cluster
