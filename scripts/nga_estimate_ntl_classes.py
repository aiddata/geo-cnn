

from __future__ import print_function, division

import os
import glob

import fiona
import rasterio
import pandas as pd
import numpy as np
import rasterio.mask

# from create_grid import PointGrid


base_path = "/sciclone/aiddata10/REU/projects/lab_oi_nigeria"





# -----------------------------------------------------------------------------
# ntl bins based on stanford science paper

class_bins = {
    0: [0, 3],
    1: [3, 35],
    2: [35, 63]
}

# -----------------------------------------------------------------------------


ntl_base = "/sciclone/aiddata10/REU/geo/data/rasters/viirs/vcmcfg_dnb_composites_v10/yearly/max"
ntl_year = 2014
ntl_path = glob.glob(os.path.join(ntl_base, "{0}_*.tif".format(ntl_year)))[0]
ntl_dim = 14


# -----------------------------------------------------------------------------


ntl_file = rasterio.open(ntl_path, 'r')

def get_ntl(lon, lat, ntl_dim=7):
    """Get nighttime lights average value for grid around point
    """
    r, c = ntl_file.index(lon, lat)
    ntl_win = ((r-ntl_dim/2+1, r+ntl_dim/2+1), (c-ntl_dim/2+1, c+ntl_dim/2+1))
    ntl_data = ntl_file.read(1, window=ntl_win)
    ntl_mean = ntl_data.mean()
    return ntl_mean


def classify(val, cat_vals):
    for cix, cval in enumerate(cat_vals):
        if val <= cval:
            return cat_names[cix]
    print(val)
    raise Exception("Could not classify")


survey_year = 2015

survey_path = os.path.join(
    base_path, "data/acled/final/acled_{}.csv".format(survey_year))

survey = pd.read_csv(survey_path, quotechar='\"',
                     na_values='', keep_default_na=False,
                     encoding='utf-8')

survey["fatalities_binary"] = (survey.fatalities > 0).astype(int)


cat_names = [0, 1]

ncats = len(cat_names)

survey_field = 'fatalities_binary'

survey_cat_vals = [0 ,1]

survey["label"] = survey[survey_field]

# # +++++++++++++++++
# # cat_names = ['low', 'med', 'high']
# cat_names = [0, 1, 2]

# ncats = len(cat_names)

# survey_field = 'fatalities'

# # determine percentiles (based on number of categories) for survey field
# survey_data_list = list(survey[survey_field])
# survey_cat_vals = [np.percentile(survey_data_list, x*100/ncats) for x in range(1, ncats+1)]

# # classify all survey locations based on percentile categories
# survey['label'] = survey.apply(
#     lambda z: classify(z[survey_field], survey_cat_vals), axis=1)
# # +++++++++++++++++


# get ntl values for each location
survey['ntl'] = survey.apply(
    lambda z: get_ntl(z['longitude'], z['latitude'], ntl_dim=ntl_dim), axis=1)

# get average ntl values by survey field categories
survey_class_ntl_mean = dict(zip(cat_names, survey.groupby('label')['ntl'].mean()))
survey_class_ntl_min = dict(zip(cat_names, survey.groupby('label')['ntl'].min()))
survey_class_ntl_max = dict(zip(cat_names, survey.groupby('label')['ntl'].max()))


print(survey_class_ntl_mean)
print(survey_class_ntl_min)
print(survey_class_ntl_max)


no_death = survey.loc[survey.label == 0].ntl
death = survey.loc[survey.label == 1].ntl

no_death_dist = [np.percentile(no_death, x) for x in np.arange(10,101,10)]
death_dist = [np.percentile(death, x) for x in np.arange(10,101,10)]

print(no_death_dist)
print(death_dist)

dl_data = []
for i in np.arange(0, 20, 0.1):
    nd_dark = np.sum(no_death < i) / len(survey)
    nd_light = np.sum(no_death > i) / len(survey)
    d_dark = np.sum(death < i) / len(survey)
    d_light = np.sum(death > i) / len(survey)
    ndd_dl = nd_dark + d_light
    dd_ndl = d_dark + nd_light
    dl_data.append((i, round(ndd_dl, 3), round(dd_ndl, 3)))

dl_columns = ["breakpoint", "ndd_dl", "dd_ndl"]

dl_df = pd.DataFrame(dl_data, columns=dl_columns)

dl_df.loc[dl_df.ndd_dl == dl_df.ndd_dl.max()]
dl_df.loc[dl_df.dd_ndl == dl_df.dd_ndl.max()]


# -----------------------------------------------------------------------------
# distribution based on pixel values in area of interest

boundary_path = os.path.join(base_path, 'data/boundary/NGA_adm0_GADM28.geojson')
boundary = fiona.open(boundary_path)


features = [feature["geometry"] for feature in boundary]

m_img, m_transform = rasterio.mask.mask(ntl_file, features, all_touched=True, crop=True)

pixel_values = m_img[(m_img.data > -9999)]

cat_vals = [np.min(pixel_values)]
cat_vals.extend([np.percentile(pixel_values, x*100/ncats) for x in range(1, ncats+1)])

print(cat_vals)
