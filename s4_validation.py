

import os
import errno
import rasterio
import pandas as pd
import numpy as np

"""
specify survey layer (with year) to validate

load s4_surface for same year (use json or fixed args to determine which surface?)

compare point (or buffer zs?) value of surface to survey value for each survey point

output dataframe of survey value, point value, buffer value along with percent differences? (include lat/lon to map errors)

"""


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def surface_win_mean(src, dim):
    r, c = src.index(row.lon, row.lat)
    win = ((r-dim/2, r+dim/2), (c-dim/2, c+dim/2))
    data = src.read(1, window=win)
    win_val = np.mean(data)
    return win_val

def sign(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


survey_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/data/surveys/final/tanzania_2015_dhs_cluster.csv"
surface_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/output/s4_surface/surface_cnn_ridge-cv10_ca5cf25_2810232_adm0_2015_v10_p10_m10_s10.tif"

survey_path = "/sciclone/aiddata10/REU/projects/mcc_ghana/data/surveys/final/ghana_2014_dhs_cluster.csv"
surface_path = "/sciclone/aiddata10/REU/projects/mcc_ghana/output/s4_surface/surface_cnn_ridge-cv10_e8b3cac_8f3999d_adm0_2014_v10_p10_m10_s10.tif"


surface_src = rasterio.open(surface_path, 'r')

survey_df = pd.read_csv(survey_path)


validation_data = []

for i, row in survey_df.iterrows():
    validation_data.append({
        "point": surface_src.sample([(row.lon, row.lat)]).next()[0],
        "dim3": surface_win_mean(surface_src, 3),
        "dim16": surface_win_mean(surface_src, 16),
        "dim33": surface_win_mean(surface_src, 33),
    })


validation_df = pd.DataFrame(validation_data)

wealthscore_sign = survey_df.wealthscore.apply(lambda x: sign(x))

print "Survey points: {}".format(len(survey_df))

for i in validation_df.columns:
    survey_df[i] = validation_df[i]
    survey_df[i+"err"] = (survey_df.wealthscore - survey_df[i]) / survey_df.wealthscore
    survey_df[i+"sign"] = (wealthscore_sign != survey_df[i].apply(lambda x: sign(x))).astype(int)
    print i
    print "\t{}".format(np.sum(survey_df[i+"sign"] ))

final_cols = ["lon", "lat", "wealthscore"]
for i in validation_df.columns:
    final_cols.append(i)
    final_cols.append(i+"err")
    final_cols.append(i+"sign")

survey_df = survey_df[final_cols]

validation_dir = os.path.dirname(os.path.dirname(surface_path)) + "/s4_validation"

validation_fname = os.path.basename(surface_path)[:-4] + "_" + os.path.basename(survey_path)

validation_path = os.path.join(validation_dir, validation_fname)

make_dir(validation_dir)
survey_df.to_csv(validation_path, index=False)


# -----------------------------------------------------------------------------

import os
import rasterio
import numpy as np
from scipy.stats.stats import pearsonr

surface_a_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/output/s4_surface/surface_cnn_ridge-cv10_ca5cf25_2810232_adm0_2010_v10_p10_m10_s10.tif"
surface_b_path = "/sciclone/aiddata10/REU/projects/mcc_tanzania/output/s4_surface/surface_cnn_ridge-cv10_ca5cf25_2810232_adm0_2015_v10_p10_m10_s10.tif"

# surface_a_path = "/sciclone/aiddata10/REU/projects/mcc_ghana/output/s4_surface/surface_cnn_ridge-cv10_e8b3cac_8f3999d_adm0_2008_v10_p10_m10_s10.tif"
# surface_b_path = "/sciclone/aiddata10/REU/projects/mcc_ghana/output/s4_surface/surface_cnn_ridge-cv10_e8b3cac_8f3999d_adm0_2014_v10_p10_m10_s10.tif"

surface_a_src = rasterio.open(surface_a_path, 'r')
surface_b_src = rasterio.open(surface_b_path, 'r')

surface_a = surface_a_src.read(1)
surface_b = surface_b_src.read(1)

flat_a = surface_a.flatten()
flat_b = surface_b.flatten()

c1 = pearsonr(flat_a, flat_b)[0]
c2 = np.corrcoef(flat_a, flat_b)[0,1]

raw_diff = surface_a - surface_b
abs_diff = np.abs(raw_diff)

validation_dir = os.path.dirname(os.path.dirname(surface_a_path)) + "/s4_validation"

raw_diff_path = os.path.join(validation_dir, "raw_diff_" + os.path.basename(surface_a_path))
abs_diff_path = os.path.join(validation_dir, "abs_diff_" + os.path.basename(surface_a_path))

meta = surface_a_src.profile

meta["crs"] = rasterio.crs.CRS.from_epsg(4236)
with rasterio.open(raw_diff_path, 'w', **meta) as result:
    result.write(np.array([raw_diff]))

with rasterio.open(abs_diff_path, 'w', **meta) as result:
    result.write(np.array([abs_diff]))
