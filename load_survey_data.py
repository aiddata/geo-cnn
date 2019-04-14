from __future__ import print_function, division

import os
import glob

import rasterio
import pandas as pd
import numpy as np

from load_data import NTL_Reader


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"

surveys = {}


# -----------------------------------------------------------------------------


ntl_calibrated = False
ntl_year = 2010
ntl_dim = 7

# ntl data
ntl = NTL_Reader(calibrated=ntl_calibrated)
ntl.set_year(ntl_year)


# -----------------------------------------------------------------------------

lsms2010_field = 'cons'

lsms2010_clusters_path = os.path.join(
    base_path, "data/surveys/final/tanzania_2010_lsms_cluster.csv")

lsms2010_cluster = pd.read_csv(lsms2010_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')

lsms2010_cluster['ntl'] = lsms2010_cluster.apply(
    lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=ntl_dim), axis=1)


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
    lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=ntl_dim), axis=1)

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
    lambda z: ntl.value(z['lon'], z['lat'], ntl_dim=ntl_dim), axis=1)

dhs2010_cluster["pred_yval"] = dhs2010_cluster[dhs2010_field]

surveys["dhs2010_cluster"] = dhs2010_cluster
