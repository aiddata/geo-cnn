from __future__ import print_function, division

import os
import glob

import rasterio
import pandas as pd
import numpy as np



base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"


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


# -----------------------------------------------------------------------------


lsms_field = 'cons'

lsms_clusters_path = os.path.join(
    base_path, "data/surveys/final/tanzania_lsms_cluster.csv")

lsms_cluster = pd.read_csv(lsms_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')

lsms_cluster['ntl'] = lsms_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)

lsms_cluster["pred_yval"] = lsms_cluster[lsms_field]


# -----------------------------------------------------------------------------


dhs_field = 'wealthscore'

dhs_clusters_path = os.path.join(
    base_path, "data/surveys/final/tanzania_dhs_cluster.csv")

dhs_cluster = pd.read_csv(dhs_clusters_path, quotechar='\"',
                          na_values='', keep_default_na=False,
                          encoding='utf-8')

dhs_cluster['ntl'] = dhs_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)

dhs_cluster["pred_yval"] = dhs_cluster[dhs_field]
