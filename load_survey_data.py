from __future__ import print_function, division

import os
import glob
import itertools
import rasterio
import pandas as pd
import numpy as np

from load_ntl_data import NTL_Reader


class SurveyData():
    """Load survey data based on static settings from settings json
    """

    def __init__(self, base_path, settings):

        self.base_path = base_path

        self.ntl_type = settings["ntl_type"]
        self.ntl_calibrated = settings["ntl_calibrated"]
        self.ntl_dim = settings["ntl_dim"]
        self.ntl_year = settings["ntl_year"]
        self.survey = settings["survey"]

        self.ntl = NTL_Reader(calibrated=self.ntl_calibrated)
        self.ntl.set_year(self.ntl_year)

        self.surveys = {}

        self._tanzania_2010_lsms_cluster()
        self._tanzania_2012_lsms_cluster()
        self._tanzania_2010_dhs_cluster()
        self._tanzania_2015_dhs_cluster()
        self._ghana_2008_dhs_cluster()
        self._ghana_2014_dhs_cluster()


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


    def _tanzania_2010_lsms_cluster(self):
        field = 'cons'

        clusters_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/data/surveys/final/tanzania_2010_lsms_cluster.csv"

        cluster = pd.read_csv(clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        cluster = self.duplicate(cluster)

        cluster['ntl'] = cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        cluster["pred_yval"] = cluster[field]

        self.surveys["tanzania_2010_lsms_cluster"] = cluster


    def _tanzania_2012_lsms_cluster(self):
        field = 'cons'

        clusters_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/data/surveys/final/tanzania_2012_lsms_cluster.csv"

        cluster = pd.read_csv(clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        cluster = self.duplicate(cluster)

        cluster['ntl'] = cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        cluster["pred_yval"] = cluster[field]

        self.surveys["tanzania_2012_lsms_cluster"] = cluster


    def _tanzania_2010_dhs_cluster(self):
        field = 'wealthscore'

        clusters_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/data/surveys/final/tanzania_2010_dhs_cluster.csv"

        cluster = pd.read_csv(clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        cluster = self.duplicate(cluster)

        cluster['ntl'] = cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        cluster["pred_yval"] = cluster[field]

        self.surveys["tanzania_2010_dhs_cluster"] = cluster


    def _tanzania_2015_dhs_cluster(self):
        field = 'wealthscore'

        clusters_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/data/surveys/final/tanzania_2015_dhs_cluster.csv"

        cluster = pd.read_csv(clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        cluster = self.duplicate(cluster)

        cluster['ntl'] = cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        cluster["pred_yval"] = cluster[field]

        self.surveys["tanzania_2015_dhs_cluster"] = cluster


    def _ghana_2008_dhs_cluster(self):
        field = 'wealthscore'

        clusters_path = "/sciclone/aiddata10/REU/projects/mcc_ghana/data/surveys/final/ghana_2008_dhs_cluster.csv"

        cluster = pd.read_csv(clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        cluster = self.duplicate(cluster)

        cluster['ntl'] = cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        cluster["pred_yval"] = cluster[field]

        self.surveys["ghana_2008_dhs_cluster"] = cluster


    def _ghana_2014_dhs_cluster(self):
        field = 'wealthscore'

        clusters_path = "/sciclone/aiddata10/REU/projects/mcc_ghana/data/surveys/final/ghana_2014_dhs_cluster.csv"

        cluster = pd.read_csv(clusters_path, quotechar='\"',
                                na_values='', keep_default_na=False,
                                encoding='utf-8')

        cluster = self.duplicate(cluster)

        cluster['ntl'] = cluster.apply(
            lambda z: self.ntl.value(z['lon'], z['lat'], ntl_dim=self.ntl_dim), axis=1)

        cluster["pred_yval"] = cluster[field]

        self.surveys["ghana_2014_dhs_cluster"] = cluster
