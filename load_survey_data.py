from __future__ import print_function, division

import os
import glob

import rasterio
import pandas as pd
import numpy as np



base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

surveys = {}


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

lsms2010_field = 'cons'

lsms2010_clusters_path = os.path.join(
    base_path, "data/surveys/final/tanzania_2010_lsms_cluster.csv")

lsms2010_cluster = pd.read_csv(lsms2010_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')

lsms2010_cluster['ntl'] = lsms2010_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)

lsms2010_cluster["pred_yval"] = lsms2010_cluster[lsms2010_field]

surveys["lsms2010_cluster"] = lsms2010_cluster

# -----------------------------------------------------------------------------


lsms2012_field = 'cons'

lsms2012_clusters_path = os.path.join(
    base_path, "data/surveys/final/tanzania_2012_lsms_cluster.csv")

lsms2012_cluster = pd.read_csv(lsms2012_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')

lsms2012_cluster['ntl'] = lsms2012_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)

lsms2012_cluster["pred_yval"] = lsms2012_cluster[lsms2012_field]

surveys["lsms2012_cluster"] = lsms2012_cluster

# -----------------------------------------------------------------------------


dhs2010_field = 'wealthscore'

dhs2010_clusters_path = os.path.join(
    base_path, "data/surveys/final/tanzania_2010_dhs_cluster.csv")

dhs2010_cluster = pd.read_csv(dhs2010_clusters_path, quotechar='\"',
                          na_values='', keep_default_na=False,
                          encoding='utf-8')

dhs2010_cluster['ntl'] = dhs2010_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)

dhs2010_cluster["pred_yval"] = dhs2010_cluster[dhs2010_field]

surveys["dhs2010_cluster"] = dhs2010_cluster
