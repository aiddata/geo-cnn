from __future__ import print_function, division

import os
import glob

import fiona
import rasterio
import pandas as pd
import numpy as np

# from create_grid import PointGrid


base_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania"


# cat_names = ['low', 'med', 'high']
cat_names = [0, 1, 2]

ncats = len(cat_names)


# -----------------------------------------------------------------------------
# ntl bins based on stanford science paper

class_bins = {
    0: [0, 3],
    1: [3, 35],
    2: [35, 63]
}

# -----------------------------------------------------------------------------


ntl_base = "/sciclone/aiddata10/REU/geo/data/rasters/dmsp_ntl/v4composites_calibrated_201709"
ntl_year = 2010
ntl_path = glob.glob(os.path.join(ntl_base, "*{0}*.tif".format(ntl_year)))[0]
ntl_file = rasterio.open(ntl_path, 'r')

def get_ntl(lon, lat, ntl_dim=7):
    """Get nighttime lights average value for grid around point
    """
    r, c = ntl_file.index(lon, lat)
    ntl_win = ((r-ntl_dim/2+1, r+ntl_dim/2+1), (c-ntl_dim/2+1, c+ntl_dim/2+1))
    ntl_data = ntl_file.read(1, window=ntl_win)
    ntl_mean = ntl_data.mean()
    return ntl_mean


# -----------------------------------------------------------------------------


def classify(val, cat_vals):
    for cix, cval in enumerate(cat_vals):
        if val <= cval:
            return cat_names[cix]
    print(val)
    raise Exception("Could not classify")

# -----------------------------------------------------------------------------


lsms_field = 'cons'

lsms_clusters_path = os.path.join(
    base_path, "data/surveys/final/tanzania_lsms_cluster.csv")

lsms_cluster = pd.read_csv(lsms_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')


# determine percentiles (based on number of categories) for survey field
lsms_data_list = list(lsms_cluster[lsms_field])
lsms_cat_vals = [np.percentile(lsms_data_list, x*100/ncats) for x in range(1, ncats+1)]

# classify all survey locations based on percentile categories
lsms_cluster['label'] = lsms_cluster.apply(
    lambda z: classify(z[lsms_field], lsms_cat_vals), axis=1)

# get ntl values for each location
lsms_cluster['ntl'] = lsms_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)

# get average ntl values by survey field categories
lsms_class_ntl_means = dict(zip(cat_names, lsms_cluster.groupby('label')['ntl'].mean()))

print(lsms_class_ntl_means)


# -----------------------------------------------------------------------------


dhs_field = 'wealthscore'

dhs_clusters_path = os.path.join(
    base_path, "data/surveys/final/tanzania_dhs_cluster.csv")

dhs_cluster = pd.read_csv(dhs_clusters_path, quotechar='\"',
                           na_values='', keep_default_na=False,
                           encoding='utf-8')

# determine percentiles (based on number of categories) for survey field
dhs_data_list = list(dhs_cluster[dhs_field])
dhs_cat_vals = [np.percentile(dhs_data_list, x*100/ncats) for x in range(1, ncats+1)]

# classify all survey locations based on percentile categories
dhs_cluster['label'] = dhs_cluster.apply(
    lambda z: classify(z[dhs_field], dhs_cat_vals), axis=1)

# get ntl values for each location
dhs_cluster['ntl'] = dhs_cluster.apply(
    lambda z: get_ntl(z['lon'], z['lat']), axis=1)

# get average ntl values by survey field categories
dhs_class_ntl_means = dict(zip(cat_names, dhs_cluster.groupby('label')['ntl'].mean()))

print(dhs_class_ntl_means)

# -----------------------------------------------------------------------------
# distribution based on pixel values in area of interest

tza_adm0_path = os.path.join(base_path, 'data/TZA_ADM0_GADM28_simplified.geojson')
tza_adm0 = fiona.open(tza_adm0_path)


features = [feature["geometry"] for feature in tza_adm0]

import rasterio.mask
m_img, m_transform = rasterio.mask.mask(ntl_file, features, all_touched=True, crop=True)

pixel_values = m_img[(m_img.data < 255)&(m_img.data > 0)]

cat_vals = [np.percentile(pixel_values, x*100/ncats) for x in range(1, ncats+1)]
